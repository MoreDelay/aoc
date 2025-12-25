use std::{collections::HashMap, num::ParseIntError, rc::Rc};

use bitvec::prelude as bv;
use rayon::prelude::*;
use thiserror::Error;

const MAX_BITS: usize = 64;
type BitArr = bv::BitArr!(for MAX_BITS);

#[derive(Debug, Copy, Clone)]
struct BitMask {
    inner: BitArr,
    used: u8,
}

impl BitMask {
    fn new(used: usize) -> Option<Self> {
        (used <= MAX_BITS).then_some(Self {
            inner: bv::bitarr![0; MAX_BITS],
            used: used as u8,
        })
    }
}

impl std::ops::Deref for BitMask {
    type Target = bv::BitSlice;

    fn deref(&self) -> &Self::Target {
        &self.inner[..self.used as usize]
    }
}

impl std::ops::DerefMut for BitMask {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner[..self.used as usize]
    }
}

impl std::ops::BitXorAssign for BitMask {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.inner ^= rhs.inner;
    }
}

#[derive(Debug, Clone, Copy)]
struct Diagram {
    target: BitMask,
}

#[derive(Debug, Error)]
pub enum DiagramParseError {
    #[error("expect format '[(.|#)+]', got '{0}'")]
    WrongFormat(String),
    #[error("can only handle {MAX_BITS} lights, but found {0}")]
    TooManyLamps(usize),
}

impl Diagram {
    fn parse(input: &str) -> Result<Self, DiagramParseError> {
        if input.len() < 3 {
            return Err(DiagramParseError::WrongFormat(input.to_string()));
        }
        let (left, input) = input.split_at(1);
        let (input, right) = input.split_at(input.len() - 1);
        match (left, right) {
            ("[", "]") => (),
            _ => return Err(DiagramParseError::WrongFormat(input.to_string())),
        }

        let lamps = input.len();
        let Some(mut target) = BitMask::new(lamps) else {
            return Err(DiagramParseError::TooManyLamps(lamps));
        };

        for (i, c) in input.chars().enumerate() {
            match c {
                '.' => (),
                '#' => target.set(i, true),
                _ => return Err(DiagramParseError::WrongFormat(input.to_string())),
            }
        }

        Ok(Self { target })
    }
}

#[derive(Debug, Clone, Copy)]
struct Button {
    flips: BitMask,
}

#[derive(Debug, Error)]
pub enum ButtonParseError {
    #[error("can only handle {MAX_BITS} lights, but provided {0}")]
    TooManyLamps(usize),
    #[error("expect format '(<number>(,<number>)*)', got '{0}'")]
    WrongFormat(String),
    #[error("could not parse '{0}' as a number")]
    NotANumber(String, #[source] ParseIntError),
    #[error("Button wants to switch lamp {0}, but only {1} lamps exist")]
    NumberTooLarge(usize, usize),
    #[error("Button flips light #{0} twice")]
    FlippedTwice(usize),
}

impl Button {
    fn parse(input: &str, lamps: usize) -> Result<Self, ButtonParseError> {
        let Some(mut flips) = BitMask::new(lamps) else {
            return Err(ButtonParseError::TooManyLamps(lamps));
        };

        if input.len() < 3 {
            return Err(ButtonParseError::WrongFormat(input.to_string()));
        }
        let (left, input) = input.split_at(1);
        let (input, right) = input.split_at(input.len() - 1);
        match (left, right) {
            ("(", ")") => (),
            _ => return Err(ButtonParseError::WrongFormat(input.to_string())),
        }

        let flip_numbers = input
            .split(',')
            .map(|v| {
                v.parse()
                    .map_err(|e| ButtonParseError::NotANumber(v.to_string(), e))
                    .and_then(|v| match v < lamps {
                        true => Ok(v),
                        false => Err(ButtonParseError::NumberTooLarge(v, lamps)),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for flip in flip_numbers {
            if flips[flip] {
                return Err(ButtonParseError::FlippedTwice(flip));
            }
            flips.set(flip, true);
        }

        Ok(Self { flips })
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct Joltage(Vec<usize>);

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct JoltageTarget(Joltage);

#[derive(Debug, Error)]
pub enum JoltageTargetParseError {
    #[error("expect format '{{<number>(,<number>)*}}', got '{0}'")]
    WrongFormat(String),
    #[error("could not parse '{0}' as a number")]
    NotANumber(String, #[source] ParseIntError),
    #[error("can only handle {MAX_BITS} lights, but found {0} joltages")]
    TooManyLamps(usize),
}

impl JoltageTarget {
    fn parse(input: &str) -> Result<Self, JoltageTargetParseError> {
        if input.len() < 3 {
            return Err(JoltageTargetParseError::WrongFormat(input.to_string()));
        }
        let (left, input) = input.split_at(1);
        let (input, right) = input.split_at(input.len() - 1);
        match (left, right) {
            ("{", "}") => (),
            _ => return Err(JoltageTargetParseError::WrongFormat(input.to_string())),
        }

        let joltages = input
            .split(',')
            .map(|v| {
                v.parse()
                    .map_err(|e| JoltageTargetParseError::NotANumber(v.to_string(), e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if joltages.len() > MAX_BITS {
            return Err(JoltageTargetParseError::TooManyLamps(joltages.len()));
        }

        Ok(Self(Joltage(joltages)))
    }

    fn as_parity_diagram(&self) -> Diagram {
        let n_lamps = self.0.0.len();
        let mut target = BitMask::new(n_lamps).expect("lamp count valid by construction");

        for (i, take) in self.0.0.iter().map(|v| (v % 2) != 0).enumerate() {
            target.set(i, take);
        }

        Diagram { target }
    }

    fn reduce_by(&self, joltage: &Joltage) -> Option<JoltageTarget> {
        let mut reduced = self.clone();
        for (reduced_v, joltage_v) in reduced.0.0.iter_mut().zip(joltage.0.iter()) {
            *reduced_v = reduced_v.checked_sub(*joltage_v)?;
        }
        Some(reduced)
    }

    fn half_reqs(self) -> Option<Self> {
        let values = self
            .0
            .0
            .into_iter()
            .map(|v| if v % 2 == 0 { Some(v / 2) } else { None })
            .collect::<Option<Vec<_>>>()?;
        Some(Self(Joltage(values)))
    }

    fn sum(&self) -> usize {
        self.0.0.iter().sum()
    }
}

#[derive(Debug)]
struct Machine {
    diagram: Diagram,
    buttons: Vec<Button>,
    joltage_target: JoltageTarget,
}

#[derive(Debug, Error)]
pub enum MachineParseError {
    #[error("missing part of machine description")]
    MissingPart,
    #[error("could not parse light diagram")]
    Diagram(#[from] DiagramParseError),
    #[error("could not parse button #{0}")]
    Button(usize, #[source] ButtonParseError),
    #[error("could not parse joltage requirements")]
    Joltages(#[from] JoltageTargetParseError),
}

impl Machine {
    fn parse(input: &str) -> Result<Self, MachineParseError> {
        let mut iter = input.split(' ');
        let diagram = iter
            .next()
            .ok_or(MachineParseError::MissingPart)
            .and_then(|s| Ok(Diagram::parse(s)?))?;
        let joltage_target = iter
            .next_back()
            .ok_or(MachineParseError::MissingPart)
            .and_then(|s| Ok(JoltageTarget::parse(s)?))?;

        let n_lamps = diagram.target.len();
        let buttons = iter
            .enumerate()
            .map(|(i, s)| Button::parse(s, n_lamps).map_err(|e| MachineParseError::Button(i, e)))
            .collect::<Result<Vec<_>, _>>()?;
        if buttons.is_empty() {
            return Err(MachineParseError::MissingPart);
        }

        Ok(Self {
            diagram,
            buttons,
            joltage_target,
        })
    }
}

#[derive(Debug, Clone)]
struct BitPatternIterator {
    active: usize,
    next: Option<BitMask>,
}

impl BitPatternIterator {
    fn new(max: usize, active: usize) -> Self {
        assert!(active <= max);

        let mut pattern =
            BitMask::new(max).expect("only allow {MAX_BITS} many bits, requested {max}");
        for i in 0..active {
            pattern.set(i, true);
        }
        Self {
            active,
            next: Some(pattern),
        }
    }
}

impl Iterator for BitPatternIterator {
    type Item = BitMask;

    fn next(&mut self) -> Option<Self::Item> {
        let res = self.next.take()?;
        let mut next = res;

        let Some(zero_pos) = next.last_zero() else {
            // special case: all bits are set, give a single element back
            return Some(res);
        };
        let slice = &next[..zero_pos];
        let Some(one_pos) = slice.last_one() else {
            // all bits right-aligned, iterator exhausted
            return Some(res);
        };

        // construct next pattern
        next.set(one_pos, false);
        let slice = &mut next[one_pos + 1..];
        let one_count = slice.count_ones() + 1;
        slice.fill_with(|i| i < one_count);
        assert!(next.count_ones() == self.active);

        self.next = Some(next);
        Some(res)
    }
}

impl Machine {
    fn make_button_mask_iterator<'a>(
        &'a self,
        diagram: Diagram,
    ) -> impl Iterator<Item = BitMask> + 'a {
        let n_buttons = self.buttons.len();
        (0..=n_buttons)
            .flat_map(move |presses| BitPatternIterator::new(n_buttons, presses))
            .filter(move |button_mask| {
                let acc = diagram.target;
                let pattern =
                    button_mask[..n_buttons]
                        .iter()
                        .enumerate()
                        .fold(acc, |mut acc, (i, take)| {
                            if *take {
                                acc ^= self.buttons[i].flips;
                            }
                            acc
                        });
                pattern.count_ones() == 0
            })
    }

    fn presses_to_boot(&self) -> Option<usize> {
        self.make_button_mask_iterator(self.diagram)
            .next()
            .map(|mask| mask.count_ones())
    }
}

struct BifurcationState<'a> {
    target: Rc<JoltageTarget>,
    mask_iter: Box<dyn Iterator<Item = BitMask> + 'a>,
    buttons_used: usize,
    solution: Option<usize>,
}

enum BifurcationOutput<'a> {
    Recurse {
        next_target: Rc<JoltageTarget>,
        state: BifurcationState<'a>,
    },
    Solved {
        target: Rc<JoltageTarget>,
        solution: Option<usize>,
    },
}

enum BifurcationInput<'a> {
    Start {
        target: Rc<JoltageTarget>,
    },
    Continue {
        state: BifurcationState<'a>,
        solution: Option<usize>,
    },
}

impl Machine {
    fn bifurcation_recursion<'a>(&'a self, input: BifurcationInput<'a>) -> BifurcationOutput<'a> {
        let state = match input {
            BifurcationInput::Start { target } => {
                if target.sum() == 0 {
                    return BifurcationOutput::Solved {
                        target,
                        solution: Some(0),
                    };
                }

                let diagram = target.as_parity_diagram();
                let mask_iter = Box::new(self.make_button_mask_iterator(diagram));
                BifurcationState {
                    target,
                    mask_iter,
                    buttons_used: 0,
                    solution: None,
                }
            }
            BifurcationInput::Continue {
                mut state,
                solution,
            } => {
                if let Some(sol) = solution {
                    let sol = state.buttons_used + 2 * sol;
                    let sol = state.solution.map(|last| last.min(sol)).or(Some(sol));
                    state.solution = sol;
                }
                state
            }
        };

        let BifurcationState {
            target,
            mut mask_iter,
            solution,
            ..
        } = state;

        let next_target = mask_iter
            .by_ref()
            .flat_map(|mask| {
                let joltage = self.joltage_from_button_mask(mask);
                let target = target.reduce_by(&joltage)?;
                let target = target.half_reqs().expect("made even above");
                let buttons_used = mask.count_ones();
                Some((target, buttons_used))
            })
            .next();

        let Some((next_target, buttons_used)) = next_target else {
            return BifurcationOutput::Solved { target, solution };
        };
        let next_target = Rc::new(next_target);

        let state = BifurcationState {
            target,
            mask_iter,
            buttons_used,
            solution,
        };
        BifurcationOutput::Recurse { next_target, state }
    }

    fn joltage_from_button_mask(&self, mask: BitMask) -> Joltage {
        let n_lamps = self.diagram.target.len();
        let mut values = vec![0; n_lamps];
        let indices = self
            .buttons
            .iter()
            .zip(&*mask)
            .filter_map(|(btn, take)| take.then_some(btn))
            .flat_map(|btn| {
                btn.flips
                    .iter()
                    .enumerate()
                    .filter_map(|(i, take)| take.then_some(i))
            });
        for i in indices {
            values[i] += 1;
        }
        Joltage(values)
    }

    fn presses_to_configure(&self) -> Option<usize> {
        use std::collections::hash_map::Entry;

        let mut cache = HashMap::new();

        let mut stack = Vec::new();
        // init stack
        {
            let target = Rc::new(self.joltage_target.clone());
            let input = BifurcationInput::Start { target };
            let output = self.bifurcation_recursion(input);
            stack.push(output);
        }

        while let Some(output) = stack.pop() {
            match output {
                BifurcationOutput::Solved { target, solution } => match cache.entry(target) {
                    Entry::Occupied(entry) => assert_eq!(*entry.get(), solution),
                    Entry::Vacant(entry) => {
                        entry.insert(solution);
                    }
                },
                BifurcationOutput::Recurse { next_target, state } => {
                    match cache.get(&next_target) {
                        Some(&solution) => {
                            let input = BifurcationInput::Continue { state, solution };
                            let output = self.bifurcation_recursion(input);
                            stack.push(output);
                        }
                        None => {
                            let input = BifurcationInput::Start {
                                target: Rc::clone(&next_target),
                            };
                            let recursive_output = self.bifurcation_recursion(input);
                            let last_output = BifurcationOutput::Recurse { next_target, state };
                            stack.push(last_output);
                            stack.push(recursive_output);
                        }
                    }
                }
            }
        }

        cache
            .get(&self.joltage_target)
            .copied()
            .expect("cache holds solution for our target")
    }
}

struct Machines(Vec<Machine>);

#[derive(Debug, Error)]
pub enum MachinesParseError {
    #[error("error while reading")]
    ReadError(#[from] std::io::Error),
    #[error("could not parse machine on line {0}")]
    Machine(usize, #[source] MachineParseError),
}

#[derive(Debug, Error)]
pub enum BootError {
    #[error("booting machine {0} is not solvable")]
    UnsolvableMachine(usize),
}

#[derive(Debug, Error)]
pub enum ConfigureError {
    #[error("configuring machine {0} is not solvable")]
    UnsolvableMachine(usize),
}

impl Machines {
    fn parse(input: impl std::io::BufRead) -> Result<Self, MachinesParseError> {
        let machines = input
            .lines()
            .enumerate()
            .map(|(i, line)| Machine::parse(&line?).map_err(|e| MachinesParseError::Machine(i, e)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self(machines))
    }

    fn presses_to_boot_all(&self) -> Result<usize, BootError> {
        self.0.iter().enumerate().try_fold(0, |acc, (i, m)| {
            let toggles = m.presses_to_boot().ok_or(BootError::UnsolvableMachine(i))?;
            Ok(acc + toggles)
        })
    }

    fn presses_to_configure_all(&self) -> Result<usize, ConfigureError> {
        self.0
            .par_iter()
            .enumerate()
            .try_fold(
                || 0,
                |acc, (i, m)| {
                    let toggles = m
                        .presses_to_configure()
                        .ok_or(ConfigureError::UnsolvableMachine(i))?;
                    Ok(acc + toggles)
                },
            )
            .try_reduce(|| 0, |a, b| Ok(a + b))
    }
}

#[derive(Debug, Error)]
pub enum Day10Error {
    #[error("could not open file")]
    FileError(#[from] std::io::Error),
    #[error("could not parse input")]
    Parsing(#[from] MachinesParseError),
    #[error("could not boot machines")]
    Boot(#[from] BootError),
    #[error("could not configure machines")]
    Configure(#[from] ConfigureError),
}

pub fn day10() -> Result<(), Day10Error> {
    let path = std::path::PathBuf::from("resources/day10.txt");
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let machines = Machines::parse(reader)?;
    let presses = machines.presses_to_boot_all()?;
    println!("day10: booted all machines with {presses} button presses");
    let presses = machines.presses_to_configure_all()?;
    println!("day10: configured all machines with {presses} button presses");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    static EXAMPLE: &str = "[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}";

    #[test]
    fn test_input_01() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let machines = Machines::parse(cursor)?;
        let presses = machines.presses_to_boot_all()?;
        assert_eq!(presses, 7);
        Ok(())
    }

    #[test]
    fn test_input_02() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let machines = Machines::parse(cursor)?;
        let presses = machines.presses_to_configure_all()?;
        assert_eq!(presses, 33);
        Ok(())
    }
}

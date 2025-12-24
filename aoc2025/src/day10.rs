use std::{collections::HashMap, num::ParseIntError};

use bitvec::prelude as bv;
use thiserror::Error;

const MAX_LAMPS: usize = 64;
type BitMask = bv::BitArr!(for MAX_LAMPS);

#[derive(Debug, Clone)]
struct Diagram {
    target: BitMask,
    lamps: usize,
}

#[derive(Debug, Error)]
pub enum DiagramParseError {
    #[error("expect format '[(.|#)+]', got '{0}'")]
    WrongFormat(String),
    #[error("can only handle {MAX_LAMPS} lights, but found {0}")]
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
        if lamps > MAX_LAMPS {
            return Err(DiagramParseError::TooManyLamps(lamps));
        }

        let mut target = bv::bitarr![0; MAX_LAMPS];
        for (i, c) in input.chars().enumerate() {
            match c {
                '.' => (),
                '#' => target.set(i, true),
                _ => return Err(DiagramParseError::WrongFormat(input.to_string())),
            }
        }

        Ok(Self { target, lamps })
    }
}

#[derive(Debug)]
struct Button {
    flips: BitMask,
}

#[derive(Debug, Error)]
pub enum ButtonParseError {
    #[error("can only handle {MAX_LAMPS} lights, but provided {0}")]
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
    fn parse(input: &str, lights: usize) -> Result<Self, ButtonParseError> {
        if lights > MAX_LAMPS {
            return Err(ButtonParseError::TooManyLamps(lights));
        }

        if input.len() < 3 {
            return Err(ButtonParseError::WrongFormat(input.to_string()));
        }
        let (left, input) = input.split_at(1);
        let (input, right) = input.split_at(input.len() - 1);
        match (left, right) {
            ("(", ")") => (),
            _ => return Err(ButtonParseError::WrongFormat(input.to_string())),
        }

        let flips = input
            .split(',')
            .map(|v| {
                v.parse()
                    .map_err(|e| ButtonParseError::NotANumber(v.to_string(), e))
                    .and_then(|v| match v < lights {
                        true => Ok(v),
                        false => Err(ButtonParseError::NumberTooLarge(v, lights)),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut button = bv::bitarr![0; MAX_LAMPS];
        for flip in flips {
            if button[flip] {
                return Err(ButtonParseError::FlippedTwice(flip));
            }
            button.set(flip, true);
        }

        Ok(Self { flips: button })
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
    #[error("can only handle {MAX_LAMPS} lights, but found {0} joltages")]
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
        if joltages.len() > MAX_LAMPS {
            return Err(JoltageTargetParseError::TooManyLamps(joltages.len()));
        }

        Ok(Self(Joltage(joltages)))
    }

    fn as_parity_diagram(&self) -> Diagram {
        let lamps = self.0.0.len();

        let mut target = bv::bitarr![0; MAX_LAMPS];
        for (i, take) in self.0.0.iter().map(|v| (v % 2) != 0).enumerate() {
            target.set(i, take);
        }

        Diagram { target, lamps }
    }

    fn reduced_with(&self, joltage: &Joltage) -> Option<JoltageTarget> {
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

        let buttons = iter
            .enumerate()
            .map(|(i, s)| {
                Button::parse(s, diagram.lamps).map_err(|e| MachineParseError::Button(i, e))
            })
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
    lamps: usize,
    active: usize,
    next: Option<BitMask>,
}

impl BitPatternIterator {
    fn new(max: usize, active: usize) -> Self {
        assert!(max < MAX_LAMPS);
        assert!(active <= max);

        let mut pattern = bv::bitarr![0; MAX_LAMPS];
        for i in 0..active {
            pattern.set(i, true);
        }
        Self {
            lamps: max,
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

        let slice = &mut next[..self.lamps];
        let Some(zero_pos) = slice.last_zero() else {
            // special case: all bits are set, give a single element back
            return Some(res);
        };
        let from_zero_slice = &slice[..zero_pos];
        let Some(one_pos) = from_zero_slice.last_one() else {
            // all bits right-aligned, iterator exhausted
            return Some(res);
        };

        // construct next pattern
        slice.set(one_pos, false);
        let replace_slice = &mut slice[one_pos + 1..];
        let one_count = replace_slice.count_ones() + 1;
        for i in 0..replace_slice.len() {
            replace_slice.set(i, i < one_count);
        }
        assert!(next.count_ones() == self.active);

        self.next = Some(next);
        Some(res)
    }
}

impl Machine {
    fn make_button_mask_iterator(&self, diagram: Diagram) -> impl Iterator<Item = BitMask> {
        (0..=self.buttons.len())
            .flat_map(|presses| BitPatternIterator::new(self.buttons.len(), presses))
            .filter(move |button_mask| {
                let acc = diagram.target;
                let pattern = button_mask
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
        self.make_button_mask_iterator(self.diagram.clone())
            .next()
            .map(|mask| mask.count_ones())
    }

    fn joltage_from_button_mask(&self, mask: BitMask) -> Joltage {
        let mut joltage = vec![0; self.diagram.lamps];
        for (button, take) in mask.into_iter().take(self.buttons.len()).enumerate() {
            if !take {
                continue;
            }
            for (i, add) in self.buttons[button].flips.iter().enumerate() {
                if *add {
                    joltage[i] += 1;
                }
            }
        }
        Joltage(joltage)
    }

    fn presses_to_configure_bifurcation(&self) -> Option<usize> {
        use std::collections::hash_map::Entry;

        struct StackState<T: Iterator<Item = BitMask>> {
            target: JoltageTarget,
            mask_iter: T,
        }

        let init = {
            let diagram = self.joltage_target.as_parity_diagram();
            let mask_iter = self.make_button_mask_iterator(diagram);
            StackState {
                target: self.joltage_target.clone(),
                mask_iter,
            }
        };
        let mut stack = Vec::from_iter([init]);

        let mut cache: HashMap<JoltageTarget, Option<usize>> = HashMap::new();
        {
            let zeros = vec![0; self.joltage_target.0.0.len()];
            let stop_condition = JoltageTarget(Joltage(zeros));
            cache.insert(stop_condition, Some(0));
        }

        'stack: while let Some(mut state) = stack.pop() {
            for mask in state.mask_iter.by_ref() {
                let joltage = self.joltage_from_button_mask(mask);
                let Some(target) = state.target.reduced_with(&joltage) else {
                    continue;
                };
                let target = target.half_reqs().expect("made all even above");
                if cache.contains_key(&target) {
                    continue;
                };

                // solve sub problem first and then return to this problem
                let diagram = target.as_parity_diagram();
                let mask_iter = self.make_button_mask_iterator(diagram);
                let new_state = StackState { target, mask_iter };
                stack.push(state);
                stack.push(new_state);
                continue 'stack;
            }

            // all sub problem solutions exist in the cache
            let diagram = state.target.as_parity_diagram();
            let mask_iter = self.make_button_mask_iterator(diagram);
            let solution = mask_iter
                .into_iter()
                .flat_map(|mask| {
                    let joltage = self.joltage_from_button_mask(mask);
                    let target = state.target.reduced_with(&joltage)?;
                    let target = target.half_reqs().expect("made all even above");
                    let solution = *cache.get(&target).expect("all solutions in cache");
                    let solution = solution? * 2 + mask.count_ones();
                    Some(solution)
                })
                .min();

            match cache.entry(state.target.clone()) {
                Entry::Occupied(entry) => assert_eq!(*entry.get(), solution),
                Entry::Vacant(entry) => {
                    entry.insert(solution);
                }
            };
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
        self.0.iter().enumerate().try_fold(0, |acc, (i, m)| {
            println!("working on machine {i}");
            let toggles = m
                .presses_to_configure_bifurcation()
                .ok_or(ConfigureError::UnsolvableMachine(i))?;
            println!("machine {i} solved with {toggles}");
            Ok(acc + toggles)
        })
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

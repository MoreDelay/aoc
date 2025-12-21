use anyhow::Result;
use std::{
    cmp::Ordering,
    fmt::{Display, Write},
    path::PathBuf,
};

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Pos(usize, usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Direction {
    Up,
    Right,
    Down,
    Left,
}

#[derive(Copy, Clone, Debug)]
struct DFSNode<T: GridGraph> {
    at: T,
    info_index: usize,
}

#[derive(Copy, Clone, Debug)]
struct RevPathInfo {
    dir: Direction,
    prev_index: usize,
}

#[derive(Clone, Debug)]
struct PathIter<T: GridGraph> {
    stack: Vec<DFSNode<T>>,
    infos: Vec<Option<RevPathInfo>>,
    goal: T,
}

impl Pos {
    fn get_dirs_towards(self, goal: Pos) -> (Option<Direction>, Option<Direction>) {
        use Direction::*;
        let Pos(sx, sy) = self;
        let Pos(gx, gy) = goal;
        let in_x = match sx.cmp(&gx) {
            Ordering::Less => Some(Right),
            Ordering::Equal => None,
            Ordering::Greater => Some(Left),
        };
        let in_y = match sy.cmp(&gy) {
            Ordering::Less => Some(Down),
            Ordering::Equal => None,
            Ordering::Greater => Some(Up),
        };
        (in_x, in_y)
    }
}

trait GridGraph: Copy + Eq + std::fmt::Debug {
    fn go_up(self) -> Result<Self, AocError>;
    fn go_right(self) -> Result<Self, AocError>;
    fn go_down(self) -> Result<Self, AocError>;
    fn go_left(self) -> Result<Self, AocError>;

    fn get_pos(self) -> Pos;

    fn go(self, dir: Direction) -> Result<Self, AocError> {
        use Direction::*;
        match dir {
            Up => self.go_up(),
            Right => self.go_right(),
            Down => self.go_down(),
            Left => self.go_left(),
        }
    }

    fn iter_paths(self, goal: Self) -> PathIter<Self> {
        let infos = vec![None];
        let initial = DFSNode {
            at: self,
            info_index: 0,
        };
        let stack = vec![initial];
        PathIter { stack, infos, goal }
    }
}

impl<T: GridGraph> Iterator for PathIter<T> {
    type Item = DirpadSequence;

    // DFS search because the graph is very constrained and this uses less memory
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            let DFSNode { at, info_index } = node;

            if at == self.goal {
                let mut rev_dirs = Vec::new();
                let mut cur_info = self.infos[info_index];
                while let Some(info) = cur_info {
                    let RevPathInfo { dir, prev_index } = info;
                    rev_dirs.push(dir);
                    cur_info = self.infos[prev_index];
                }
                rev_dirs.reverse();
                return Some(rev_dirs.into());
            }

            // only choose neighbor nodes that lead towards the goal
            let at_pos = at.get_pos();
            let goal_pos = self.goal.get_pos();
            let (in_x, in_y) = at_pos.get_dirs_towards(goal_pos);

            if let Some(dir) = in_x {
                if let Ok(next_at) = at.go(dir) {
                    let path_info = RevPathInfo {
                        dir,
                        prev_index: info_index,
                    };
                    self.infos.push(Some(path_info));
                    let next_index = self.infos.len() - 1;
                    let next = DFSNode {
                        at: next_at,
                        info_index: next_index,
                    };
                    self.stack.push(next);
                }
            }

            if let Some(dir) = in_y {
                if let Ok(next_at) = at.go(dir) {
                    let path_info = RevPathInfo {
                        dir,
                        prev_index: info_index,
                    };
                    self.infos.push(Some(path_info));
                    let next_index = self.infos.len() - 1;
                    let next = DFSNode {
                        at: next_at,
                        info_index: next_index,
                    };
                    self.stack.push(next);
                }
            }
        }
        None
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NumpadButton {
    ButtonA,
    Button0,
    Button1,
    Button2,
    Button3,
    Button4,
    Button5,
    Button6,
    Button7,
    Button8,
    Button9,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DirpadButton {
    ButtonA,
    ButtonU,
    ButtonR,
    ButtonD,
    ButtonL,
}

impl NumpadButton {
    fn parse(c: char) -> Result<Self> {
        use NumpadButton::*;
        match c {
            'A' => Ok(ButtonA),
            '0' => Ok(Button0),
            '1' => Ok(Button1),
            '2' => Ok(Button2),
            '3' => Ok(Button3),
            '4' => Ok(Button4),
            '5' => Ok(Button5),
            '6' => Ok(Button6),
            '7' => Ok(Button7),
            '8' => Ok(Button8),
            '9' => Ok(Button9),
            _ => Err(AocError::ParseError.into()),
        }
    }
}

impl Display for NumpadButton {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use NumpadButton::*;
        match self {
            ButtonA => f.write_char('A'),
            Button0 => f.write_char('0'),
            Button1 => f.write_char('1'),
            Button2 => f.write_char('2'),
            Button3 => f.write_char('3'),
            Button4 => f.write_char('4'),
            Button5 => f.write_char('5'),
            Button6 => f.write_char('6'),
            Button7 => f.write_char('7'),
            Button8 => f.write_char('8'),
            Button9 => f.write_char('9'),
        }
    }
}

impl Display for DirpadButton {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use DirpadButton::*;
        match self {
            ButtonA => f.write_char('A'),
            ButtonU => f.write_char('^'),
            ButtonR => f.write_char('>'),
            ButtonD => f.write_char('v'),
            ButtonL => f.write_char('<'),
        }
    }
}

impl GridGraph for NumpadButton {
    fn go_up(self) -> Result<Self, AocError> {
        use AocError::ValueError;
        use NumpadButton::*;
        match self {
            ButtonA => Ok(Button3),
            Button0 => Ok(Button2),
            Button1 => Ok(Button4),
            Button2 => Ok(Button5),
            Button3 => Ok(Button6),
            Button4 => Ok(Button7),
            Button5 => Ok(Button8),
            Button6 => Ok(Button9),
            Button7 => Err(ValueError("no more button".into())),
            Button8 => Err(ValueError("no more button".into())),
            Button9 => Err(ValueError("no more button".into())),
        }
    }

    fn go_right(self) -> Result<Self, AocError> {
        use AocError::ValueError;
        use NumpadButton::*;
        match self {
            ButtonA => Err(ValueError("no more button".into())),
            Button0 => Ok(ButtonA),
            Button1 => Ok(Button2),
            Button2 => Ok(Button3),
            Button3 => Err(ValueError("no more button".into())),
            Button4 => Ok(Button5),
            Button5 => Ok(Button6),
            Button6 => Err(ValueError("no more button".into())),
            Button7 => Ok(Button8),
            Button8 => Ok(Button9),
            Button9 => Err(ValueError("no more button".into())),
        }
    }

    fn go_down(self) -> Result<Self, AocError> {
        use AocError::ValueError;
        use NumpadButton::*;
        match self {
            ButtonA => Err(ValueError("no more button".into())),
            Button0 => Err(ValueError("no more button".into())),
            Button1 => Err(ValueError("no more button".into())),
            Button2 => Ok(Button0),
            Button3 => Ok(ButtonA),
            Button4 => Ok(Button1),
            Button5 => Ok(Button2),
            Button6 => Ok(Button3),
            Button7 => Ok(Button4),
            Button8 => Ok(Button5),
            Button9 => Ok(Button6),
        }
    }

    fn go_left(self) -> Result<Self, AocError> {
        use AocError::ValueError;
        use NumpadButton::*;
        match self {
            ButtonA => Ok(Button0),
            Button0 => Err(ValueError("no more button".into())),
            Button1 => Err(ValueError("no more button".into())),
            Button2 => Ok(Button1),
            Button3 => Ok(Button2),
            Button4 => Err(ValueError("no more button".into())),
            Button5 => Ok(Button4),
            Button6 => Ok(Button5),
            Button7 => Err(ValueError("no more button".into())),
            Button8 => Ok(Button7),
            Button9 => Ok(Button8),
        }
    }

    fn get_pos(self) -> Pos {
        use NumpadButton::*;
        match self {
            ButtonA => Pos(2, 3),
            Button0 => Pos(1, 3),
            Button1 => Pos(0, 2),
            Button2 => Pos(1, 2),
            Button3 => Pos(2, 2),
            Button4 => Pos(0, 1),
            Button5 => Pos(1, 1),
            Button6 => Pos(2, 1),
            Button7 => Pos(0, 0),
            Button8 => Pos(1, 0),
            Button9 => Pos(2, 0),
        }
    }
}

impl GridGraph for DirpadButton {
    fn go_up(self) -> Result<Self, AocError> {
        use AocError::ValueError;
        use DirpadButton::*;
        match self {
            ButtonA => Err(ValueError("no more button".into())),
            ButtonU => Err(ValueError("no more button".into())),
            ButtonR => Ok(ButtonA),
            ButtonD => Ok(ButtonU),
            ButtonL => Err(ValueError("no more button".into())),
        }
    }

    fn go_right(self) -> Result<Self, AocError> {
        use AocError::ValueError;
        use DirpadButton::*;
        match self {
            ButtonA => Err(ValueError("no more button".into())),
            ButtonU => Ok(ButtonA),
            ButtonR => Err(ValueError("no more button".into())),
            ButtonD => Ok(ButtonR),
            ButtonL => Ok(ButtonD),
        }
    }

    fn go_down(self) -> Result<Self, AocError> {
        use AocError::ValueError;
        use DirpadButton::*;
        match self {
            ButtonA => Ok(ButtonR),
            ButtonU => Ok(ButtonD),
            ButtonR => Err(ValueError("no more button".into())),
            ButtonD => Err(ValueError("no more button".into())),
            ButtonL => Err(ValueError("no more button".into())),
        }
    }

    fn go_left(self) -> Result<Self, AocError> {
        use AocError::ValueError;
        use DirpadButton::*;
        match self {
            ButtonA => Ok(ButtonU),
            ButtonU => Err(ValueError("no more button".into())),
            ButtonR => Ok(ButtonD),
            ButtonD => Ok(ButtonL),
            ButtonL => Err(ValueError("no more button".into())),
        }
    }

    fn get_pos(self) -> Pos {
        use DirpadButton::*;
        match self {
            ButtonA => Pos(2, 0),
            ButtonU => Pos(1, 0),
            ButtonR => Pos(2, 1),
            ButtonD => Pos(1, 1),
            ButtonL => Pos(0, 1),
        }
    }
}

#[derive(Debug, Clone)]
struct NumpadSequence(Vec<NumpadButton>);
#[derive(Debug, Clone)]
struct DirpadSequence(Vec<DirpadButton>);

impl Display for NumpadSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // skip initial A
        self.0.iter().skip(1).try_for_each(|b| b.fmt(f))
    }
}

impl Display for DirpadSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // skip initial A
        self.0.iter().skip(1).try_for_each(|b| b.fmt(f))
    }
}

impl From<Vec<Direction>> for DirpadSequence {
    fn from(value: Vec<Direction>) -> Self {
        use Direction::*;
        use DirpadButton::*;

        let mut vec = Vec::with_capacity(value.len() + 2);
        // every sequence starts at A
        vec.push(ButtonA);

        for dir in value.into_iter() {
            let b = match dir {
                Up => ButtonU,
                Right => ButtonR,
                Down => ButtonD,
                Left => ButtonL,
            };
            vec.push(b);
        }
        // to confirm the action, we need to activate after moving
        // this is also the reset for the next sequence
        vec.push(ButtonA);
        DirpadSequence(vec)
    }
}

impl NumpadSequence {
    fn parse(input: &str) -> Result<Self> {
        use NumpadButton::*;
        let mut vec = Vec::with_capacity(input.len() + 2);
        // every sequence starts at A
        vec.push(ButtonA);
        for b in input.chars().map(NumpadButton::parse) {
            vec.push(b?);
        }
        Ok(Self(vec))
    }

    fn code_value(&self) -> usize {
        let s = self.to_string();
        let s = s.strip_suffix("A").expect("all sequences end with A");
        s.parse().expect("only single A per sequence")
    }
}

fn get_button_count(num_seq: &NumpadSequence, indirection: usize) -> usize {
    use DirpadButton::*;
    const ID_MAPPING: [DirpadButton; 5] = [ButtonA, ButtonU, ButtonR, ButtonD, ButtonL];
    const START_GOAL_COMBIS: usize = ID_MAPPING.len() * ID_MAPPING.len();

    let mut table = vec![[0; START_GOAL_COMBIS]; indirection + 1];
    for i in 0..25 {
        table[0][i] = 1;
    }

    fn get_path_cost(path: &[DirpadButton], move_cost: &[usize]) -> usize {
        path.windows(2)
            .flat_map(<&[DirpadButton; 2]>::try_from)
            .map(|&[s, g]| {
                let s_id = ID_MAPPING
                    .iter()
                    .position(|&b| b == s)
                    .expect("all buttons in mapping");
                let g_id = ID_MAPPING
                    .iter()
                    .position(|&b| b == g)
                    .expect("all buttons in mapping");
                let index = s_id * ID_MAPPING.len() + g_id;
                move_cost[index]
            })
            .sum()
    }

    // create all point to point costs for all levels starting from user
    for level in 1..=indirection {
        for (start_id, &start) in ID_MAPPING.iter().enumerate() {
            for (goal_id, &goal) in ID_MAPPING.iter().enumerate() {
                let min_cost = start
                    .iter_paths(goal)
                    .map(|DirpadSequence(path)| get_path_cost(&path, &table[level - 1]))
                    .min()
                    .expect("at least one path exists");

                let index = start_id * ID_MAPPING.len() + goal_id;
                table[level][index] = min_cost;
            }
        }
    }

    // final movement taken decided by numpad sequence
    let NumpadSequence(num_seq) = num_seq.clone();

    num_seq
        .windows(2)
        .flat_map(<&[NumpadButton; 2]>::try_from)
        .map(|&[start, goal]| {
            start
                .iter_paths(goal)
                .map(|DirpadSequence(path)| get_path_cost(&path, &table[indirection]))
                .min()
                .expect("at least one path exists")
        })
        .sum()
}

fn compute_total_complexity(seq: &NumpadSequence, indirection: usize) -> usize {
    let code_value = seq.code_value();
    let button_count = get_button_count(seq, indirection);
    code_value * button_count
}

fn parse_sequences(input: &str) -> Result<Vec<NumpadSequence>> {
    input
        .split("\n")
        .filter(|s| !s.is_empty())
        .map(NumpadSequence::parse)
        .collect()
}

pub fn run() -> Result<()> {
    println!("day 21");
    let path = PathBuf::from("./resources/day21.txt");
    let data = util::get_data_string(&path)?;
    let sequences = parse_sequences(&data)?;
    let with_few_indirections: usize = sequences
        .iter()
        .map(|s| compute_total_complexity(s, 2))
        .sum();
    println!("sum of complexities with 3 robots: {with_few_indirections}");
    let with_many_indirections: usize = sequences
        .iter()
        .map(|s| compute_total_complexity(s, 25))
        .sum();
    println!("sum of complexities with 26 robots: {with_many_indirections}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_complexity() {
        let input = "029A
980A
179A
456A
379A";
        let sequences = parse_sequences(input).unwrap();
        let total_complexity: usize = sequences
            .iter()
            .map(|s| compute_total_complexity(s, 2))
            .sum();
        assert_eq!(total_complexity, 126384);
    }

    #[test]
    fn test_path_iter_on_numpad() {
        let start = NumpadButton::ButtonA;
        let goal = NumpadButton::Button1;
        let iter = start.iter_paths(goal);
        assert_eq!(iter.count(), 2);
    }
}

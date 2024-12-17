use anyhow::Result;
use core::panic;
use std::{collections::BinaryHeap, path::PathBuf};

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Pos(usize, usize);
#[derive(Copy, Clone, Debug)]
struct Movement(usize, usize);

impl Movement {
    fn parse(input: &str) -> Result<Movement, AocError> {
        let Some((x, y)) = input.split_once(", ") else {
            return Err(AocError::ParseError);
        };
        let x = x.strip_prefix("X+").ok_or(AocError::ParseError)?;
        let x = x.parse().map_err(|_| AocError::ParseError)?;
        let y = y.strip_prefix("Y+").ok_or(AocError::ParseError)?;
        let y = y.parse().map_err(|_| AocError::ParseError)?;
        let movement = Movement(x, y);
        Ok(movement)
    }
}

#[derive(Copy, Clone, Debug)]
struct Machine {
    button_a: Movement,
    button_b: Movement,
    prize: Pos,
}

fn parse_machines(input: &str) -> Result<Vec<Machine>> {
    let rows = input
        .split("\n")
        .filter(|s| s.len() != 0)
        .collect::<Vec<_>>();
    let machines = rows
        .chunks(3)
        .flat_map(<&[&str; 3]>::try_from)
        .map(|&[a, b, p]| {
            let button_a = a.strip_prefix("Button A: ").ok_or(AocError::ParseError)?;
            let button_b = b.strip_prefix("Button B: ").ok_or(AocError::ParseError)?;
            let button_a = Movement::parse(button_a)?;
            let button_b = Movement::parse(button_b)?;

            let p = p.strip_prefix("Prize: ").ok_or(AocError::ParseError)?;
            let Some((x, y)) = p.split_once(", ") else {
                return Err(AocError::ParseError.into());
            };
            let x = x.strip_prefix("X=").ok_or(AocError::ParseError)?;
            let x = x.parse().map_err(|_| AocError::ParseError)?;
            let y = y.strip_prefix("Y=").ok_or(AocError::ParseError)?;
            let y = y.parse().map_err(|_| AocError::ParseError)?;
            let prize = Pos(x, y);
            Ok(Machine {
                button_a,
                button_b,
                prize,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(machines)
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Node {
    tokens: usize,
    pos: Pos,
    used_a: usize,
    used_b: usize,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.tokens.cmp(&self.tokens)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn do_best_play(machine: Machine) -> Option<usize> {
    let Machine {
        button_a,
        button_b,
        prize,
    } = machine;
    let Pos(prize_x, prize_y) = prize;

    let mut priority_queue = BinaryHeap::new();
    let initial_node = Node {
        tokens: 0,
        pos: Pos(0, 0),
        used_a: 0,
        used_b: 0,
    };
    priority_queue.push(initial_node);

    const MAX_ITER: usize = 101 * 101;
    let mut iter = 0;
    while !priority_queue.is_empty() {
        if iter > MAX_ITER {
            panic!("too many iterations");
        }
        iter += 1;

        let top = priority_queue.pop().expect("checked not empty");
        let Node {
            tokens,
            pos,
            used_a,
            used_b,
        } = top;

        if pos == prize {
            return Some(tokens);
        }
        let Pos(x, y) = pos;
        if x > prize_x || y > prize_y {
            continue;
        }

        if used_b == 0 && used_a < 100 {
            let Movement(add_x, add_y) = button_a;
            let new_pos = Pos(x + add_x, y + add_y);
            let use_a = Node {
                tokens: tokens + 3,
                pos: new_pos,
                used_a: used_a + 1,
                used_b,
            };
            priority_queue.push(use_a);
        }
        if used_b < 100 {
            let Movement(add_x, add_y) = button_b;
            let new_pos = Pos(x + add_x, y + add_y);
            let use_b = Node {
                tokens: tokens + 1,
                pos: new_pos,
                used_a,
                used_b: used_b + 1,
            };
            priority_queue.push(use_b);
        }
    }
    None
}

fn find_optimal_cost(machines: &Vec<Machine>) -> (usize, usize) {
    let mut tokens = 0;
    let mut prizes = 0;
    for &machine in machines {
        if let Some(cost) = do_best_play(machine) {
            tokens += cost;
            prizes += 1;
        };
    }
    (tokens, prizes)
}

pub fn run() -> Result<()> {
    println!("day 13");
    let path = PathBuf::from("./resources/day13.txt");
    let data = util::get_data_string(&path)?;
    let machines = parse_machines(&data)?;
    let (tokens, prizes) = find_optimal_cost(&machines);
    println!("need {tokens} to win {prizes} prizes");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winnable_machine() {
        let input = "Button A: X+94, Y+34
Button B: X+22, Y+67
Prize: X=8400, Y=5400";
        let machine = parse_machines(input).unwrap();
        let (tokens, prizes) = find_optimal_cost(&machine);
        assert_eq!(prizes, 1);
        assert_eq!(tokens, 280);
    }

    #[test]
    fn test_unwinnable_machine() {
        let input = "Button A: X+26, Y+66
Button B: X+67, Y+21
Prize: X=12748, Y=12176";
        let machine = parse_machines(input).unwrap();
        let (tokens, prizes) = find_optimal_cost(&machine);
        assert_eq!(prizes, 0);
        assert_eq!(tokens, 0);
    }

    #[test]
    fn test_two() {
        let input = "Button A: X+94, Y+34
Button B: X+22, Y+67
Prize: X=8400, Y=5400

Button A: X+26, Y+66
Button B: X+67, Y+21
Prize: X=12748, Y=12176";
        let machines = parse_machines(input).unwrap();
        let (tokens, prizes) = find_optimal_cost(&machines);
        assert_eq!(prizes, 1);
        assert_eq!(tokens, 280);
    }
}

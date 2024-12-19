use anyhow::Result;
use core::panic;
use std::{collections::BinaryHeap, path::PathBuf};

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Pos(usize, usize);
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

fn correct_machines(mut machines: Vec<Machine>) -> Vec<Machine> {
    for machine in machines.iter_mut() {
        let Pos(x, y) = &mut machine.prize;
        *x += 10000000000000;
        *y += 10000000000000;
    }
    machines
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
fn find_optimal_cost_dijkstra(machines: &Vec<Machine>) -> (usize, usize) {
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

fn check_steps_close_to(
    machine: Machine,
    expensive_steps: usize,
    cheap_steps: usize,
) -> Option<(usize, usize)> {
    let Machine {
        button_a,
        button_b,
        prize,
    } = machine;
    let Movement(expensive_x, expensive_y) = button_a;
    let Movement(cheap_x, cheap_y) = button_b;

    for i in 0..3 {
        if i == 0 && cheap_steps == 0 {
            continue;
        };
        let test_cheap = cheap_steps + i - 1;

        for j in 0..3 {
            if j == 0 && expensive_steps == 0 {
                continue;
            };
            let test_expensive = expensive_steps + j - 1;

            let got_x = test_cheap * cheap_x + test_expensive * expensive_x;
            let got_y = test_cheap * cheap_y + test_expensive * expensive_y;
            let got = Pos(got_x, got_y);

            if got == prize {
                return Some((test_expensive, test_cheap));
            }
        }
    }
    return None;
}

fn check_machine(machine: Machine) -> Option<usize> {
    let Machine {
        button_a,
        button_b,
        prize,
    } = machine;
    let Movement(expensive_x, expensive_y) = button_a;
    let Movement(cheap_x, cheap_y) = button_b;
    let Pos(target_x, target_y) = prize;

    // edge case: both buttons move the same
    if button_a == button_b {
        // solve i * ax = tx
        // solve i * ay = ty
        let ax = cheap_x as f64;
        let bx = cheap_y as f64;
        let tx = target_x as f64;
        let ty = target_y as f64;
        let i = tx / ax;
        let j = ty / bx;

        let i = i.round() as usize;
        let j = j.round() as usize;
        if i.abs_diff(j) > 1 {
            return None;
        }
        let Some((j, i)) = check_steps_close_to(machine, 0, i) else {
            return None;
        };
        return Some(j * 3 + i);
    }

    // at least one button does not move diagonal
    // solve i * ax + j * bx = tx
    // solve i * ay + j * by = ty
    let ax = cheap_x as f64;
    let ay = cheap_y as f64;
    let bx = expensive_x as f64;
    let by = expensive_y as f64;
    let tx = target_x as f64;
    let ty = target_y as f64;

    // 0      + j * by = ty
    let by = by - (ay / ax) * bx;
    let ty = ty - (ay / ax) * tx;

    // 0      + j      = ty
    let ty = ty / by;

    // i * ax + 0      = tx
    let tx = tx - bx * ty;

    // i               = tx
    let tx = tx / ax;

    let cheap_steps = tx.round() as usize;
    let expensive_steps = ty.round() as usize;

    let Some((j, i)) = check_steps_close_to(machine, expensive_steps, cheap_steps) else {
        return None;
    };
    Some(j * 3 + i)
}

fn find_optimal_cost_equation(machines: &Vec<Machine>) -> (usize, usize) {
    let mut tokens = 0;
    let mut prizes = 0;
    for &machine in machines {
        if let Some(cost) = check_machine(machine) {
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
    // let (tokens, prizes) = find_optimal_cost_dijkstra(&machines);
    let (tokens, prizes) = find_optimal_cost_equation(&machines);
    println!("need {tokens} to win {prizes} prizes");
    let machines = correct_machines(machines);
    let (tokens, prizes) = find_optimal_cost_equation(&machines);
    println!("after correction, need {tokens} to win {prizes} prizes");
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
        let (tokens, prizes) = find_optimal_cost_dijkstra(&machine);
        assert_eq!(prizes, 1);
        assert_eq!(tokens, 280);
    }

    #[test]
    fn test_unwinnable_machine() {
        let input = "Button A: X+26, Y+66
Button B: X+67, Y+21
Prize: X=12748, Y=12176";
        let machine = parse_machines(input).unwrap();
        let (tokens, prizes) = find_optimal_cost_dijkstra(&machine);
        assert_eq!(prizes, 0);
        assert_eq!(tokens, 0);
    }

    #[test]
    fn test_two_machines() {
        let input = "Button A: X+94, Y+34
Button B: X+22, Y+67
Prize: X=8400, Y=5400

Button A: X+26, Y+66
Button B: X+67, Y+21
Prize: X=12748, Y=12176";
        let machines = parse_machines(input).unwrap();
        let (tokens, prizes) = find_optimal_cost_dijkstra(&machines);
        assert_eq!(prizes, 1);
        assert_eq!(tokens, 280);
    }

    #[test]
    fn test_far_prize_1() {
        let input = "Button A: X+94, Y+34
Button B: X+22, Y+67
Prize: X=8400, Y=5400";
        let machine = parse_machines(input).unwrap();
        let machine = correct_machines(machine);
        let (_tokens, prizes) = find_optimal_cost_equation(&machine);
        assert_eq!(prizes, 0);
    }

    #[test]
    fn test_far_prize_2() {
        let input = "Button A: X+26, Y+66
Button B: X+67, Y+21
Prize: X=12748, Y=12176";
        let machine = parse_machines(input).unwrap();
        let machine = correct_machines(machine);
        let (_tokens, prizes) = find_optimal_cost_equation(&machine);
        assert_eq!(prizes, 1);
    }

    #[test]
    fn test_far_prize_3() {
        let input = "Button A: X+17, Y+86
Button B: X+84, Y+37
Prize: X=7870, Y=6450";
        let machine = parse_machines(input).unwrap();
        let machine = correct_machines(machine);
        let (_tokens, prizes) = find_optimal_cost_equation(&machine);
        assert_eq!(prizes, 0);
    }

    #[test]
    fn test_far_prize_4() {
        let input = "Button A: X+69, Y+23
Button B: X+27, Y+71
Prize: X=18641, Y=10279";
        let machine = parse_machines(input).unwrap();
        let machine = correct_machines(machine);
        let (_tokens, prizes) = find_optimal_cost_equation(&machine);
        assert_eq!(prizes, 1);
    }

    #[test]
    fn test_far_prize_same_as_old_solution() {
        let path = PathBuf::from("./resources/day13.txt");
        let data = util::get_data_string(&path).unwrap();
        let machines = parse_machines(&data).unwrap();
        let (tokens, prizes) = find_optimal_cost_dijkstra(&machines);
        let (tokens_q, prizes_q) = find_optimal_cost_equation(&machines);
        assert_eq!(prizes, prizes_q);
        assert_eq!(tokens, tokens_q);
    }
}

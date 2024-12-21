use anyhow::Result;
use std::{char, fmt::Write, path::PathBuf};

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Pos(usize, usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Velocity(isize, isize);

#[derive(Copy, Clone, Debug)]
struct Robot {
    pos: Pos,
    vel: Velocity,
}

#[derive(Clone)]
struct Map {
    robots: Vec<Robot>,
    size: Pos,
}

impl Robot {
    fn step(&mut self, max: Pos) {
        let Pos(px, py) = self.pos;
        let Velocity(vx, vy) = self.vel;
        let Pos(mx, my) = max;
        self.pos = Pos(
            (px + (mx as isize + vx) as usize) % mx,
            (py + (my as isize + vy) as usize) % my,
        );
    }
}

impl std::fmt::Display for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Pos(mx, my) = self.size;
        for y in 0..my {
            for x in 0..mx {
                let cur = Pos(x, y);
                let count = self.robots.iter().filter(|r| r.pos == cur).count();
                if count > 0 {
                    f.write_char(char::from_digit(count as u32, 10).ok_or(std::fmt::Error)?)?;
                } else {
                    f.write_char('.')?;
                }
            }
            if y < my - 1 {
                f.write_char('\n')?;
            }
        }
        Ok(())
    }
}

fn parse_robo_map(input: &str, size: Pos) -> Result<Map> {
    let robots = input
        .split("\n")
        .filter(|s| !s.is_empty())
        .map(|s| {
            let Some((pos, vel)) = s.trim().split_once(" ") else {
                return Err(AocError::ParseError.into());
            };
            let pos = pos.strip_prefix("p=").ok_or(AocError::ParseError)?;
            let Some((px, py)) = pos.split_once(",") else {
                return Err(AocError::ParseError.into());
            };
            let px = px.parse().map_err(|_| AocError::ParseError)?;
            let py = py.parse().map_err(|_| AocError::ParseError)?;
            let pos = Pos(px, py);
            if px >= 101 || py >= 103 {
                return Err(AocError::ParseError.into());
            }

            let vel = vel.strip_prefix("v=").ok_or(AocError::ParseError)?;
            let Some((vx, vy)) = vel.split_once(",") else {
                return Err(AocError::ParseError.into());
            };
            let vx = vx.parse().map_err(|_| AocError::ParseError)?;
            let vy = vy.parse().map_err(|_| AocError::ParseError)?;
            let vel = Velocity(vx, vy);

            Ok(Robot { pos, vel })
        })
        .collect::<Result<_>>()?;
    Ok(Map { robots, size })
}

fn simulate_steps(mut map: Map, steps: usize) -> Map {
    for _step in 0..steps {
        map.robots.iter_mut().for_each(|r| r.step(map.size));
    }
    map
}

fn calculate_safety_factor(map: &Map) -> usize {
    let mut upper_left = 0;
    let mut upper_right = 0;
    let mut lower_left = 0;
    let mut lower_right = 0;

    for robot in map.robots.iter() {
        let Pos(x, y) = robot.pos;
        let Pos(mx, my) = map.size;
        let mx = mx / 2;
        let my = my / 2;
        if x < mx && y < my {
            upper_left += 1;
        }
        if x > mx && y < my {
            upper_right += 1;
        }
        if x < mx && y > my {
            lower_left += 1;
        }
        if x > mx && y > my {
            lower_right += 1;
        }
    }

    upper_left * upper_right * lower_left * lower_right
}

#[allow(dead_code)]
fn search_for_christmas_tree(mut map: Map) {
    println!("Search for the christmas tree by progressing step by step!");
    let mut buf = String::new();
    map = simulate_steps(map, 23);
    for step in (23..).step_by(101) {
        println!("step {step}:");
        println!("{map}");
        match std::io::stdin().read_line(&mut buf) {
            Err(_) => break,
            Ok(n) if n != 1 => break,
            Ok(_) => (),
        }
        map = simulate_steps(map, 101);
    }
}

pub fn run() -> Result<()> {
    println!("day 14");
    let path = PathBuf::from("./resources/day14.txt");
    let data = util::get_data_string(&path)?;
    let max = Pos(101, 103);
    let map = parse_robo_map(&data, max)?;
    let map_sim = simulate_steps(map.clone(), 100);
    let safety_factor = calculate_safety_factor(&map_sim);
    println!("safety factor: {safety_factor}");
    // search_for_christmas_tree(map);
    let until_tree = 7093;
    let map_tree = simulate_steps(map, until_tree);
    println!("at step {until_tree} we find the christmas tree:");
    println!("{map_tree}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robot_movement() {
        let mut robot = Robot {
            pos: Pos(0, 0),
            vel: Velocity(-1, -1),
        };
        robot.step(Pos(101, 103));
        assert_eq!(robot.pos, Pos(100, 102));
    }

    #[test]
    fn test_example() {
        let s = "p=0,4 v=3,-3
p=6,3 v=-1,-3
p=10,3 v=-1,2
p=2,0 v=2,-1
p=0,0 v=1,3
p=3,0 v=-2,-2
p=7,6 v=-1,-3
p=3,0 v=-1,-2
p=9,3 v=2,3
p=7,3 v=-1,2
p=2,4 v=2,-3
p=9,5 v=-3,-3";
        let max = Pos(11, 7);
        let robots = parse_robo_map(s, max).unwrap();
        let robots = simulate_steps(robots, 100);
        let factor = calculate_safety_factor(&robots);
        assert_eq!(factor, 12);
    }
}

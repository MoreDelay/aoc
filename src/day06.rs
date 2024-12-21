use anyhow::Result;
use std::{fmt::Write, ops::BitOr, path::PathBuf};

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug, PartialEq)]
enum Direction {
    Up,
    Right,
    Down,
    Left,
}

impl TryFrom<char> for Direction {
    type Error = AocError;

    fn try_from(value: char) -> std::result::Result<Self, Self::Error> {
        match value {
            '^' => Ok(Direction::Up),
            '>' => Ok(Direction::Right),
            'v' => Ok(Direction::Down),
            '<' => Ok(Direction::Left),
            _ => Err(AocError::ParseError),
        }
    }
}

impl Direction {
    fn looking_at(&self, x: usize, y: usize) -> Option<(usize, usize)> {
        Some(match *self {
            Direction::Up => (x, y.checked_sub(1)?),
            Direction::Right => (x.checked_add(1)?, y),
            Direction::Down => (x, y.checked_add(1)?),
            Direction::Left => (x.checked_sub(1)?, y),
        })
    }
}

#[derive(Copy, Clone, Debug)]
struct Guard(Direction);

impl Guard {
    fn turn(self) -> Self {
        Self(match self.0 {
            Direction::Up => Direction::Right,
            Direction::Right => Direction::Down,
            Direction::Down => Direction::Left,
            Direction::Left => Direction::Up,
        })
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct Visited {
    up: bool,
    right: bool,
    down: bool,
    left: bool,
}

impl BitOr for Visited {
    type Output = Visited;

    fn bitor(self, rhs: Self) -> Self::Output {
        Visited {
            up: self.up | rhs.up,
            right: self.right | rhs.right,
            down: self.down | rhs.down,
            left: self.left | rhs.left,
        }
    }
}

impl Visited {
    fn new(dir: &Direction) -> Self {
        match dir {
            Direction::Up => Visited {
                up: true,
                ..Default::default()
            },
            Direction::Right => Visited {
                right: true,
                ..Default::default()
            },
            Direction::Down => Visited {
                down: true,
                ..Default::default()
            },
            Direction::Left => Visited {
                left: true,
                ..Default::default()
            },
        }
    }

    fn matches_direction(&self, dir: &Direction) -> bool {
        let Visited {
            up,
            right,
            down,
            left,
        } = *self;
        match dir {
            Direction::Up => up,
            Direction::Right => right,
            Direction::Down => down,
            Direction::Left => left,
        }
    }
}

#[derive(Copy, Clone)]
enum Tile {
    Empty,
    Visited(Visited),
    Obstacle,
    Guard(Guard, Visited),
}

impl std::fmt::Debug for Tile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Tile::Empty => f.write_char('.'),
            Tile::Visited(Visited {
                up,
                right,
                down,
                left,
            }) => {
                let vertical = up | down;
                let horizontal = left | right;
                match (horizontal, vertical) {
                    (true, true) => f.write_char('+'),
                    (false, true) => f.write_char('|'),
                    (true, false) => f.write_char('-'),
                    _ => unreachable!(),
                }
            }
            Tile::Obstacle => f.write_char('#'),
            Tile::Guard(guard, _) => match guard.0 {
                Direction::Up => f.write_char('^'),
                Direction::Right => f.write_char('>'),
                Direction::Down => f.write_char('v'),
                Direction::Left => f.write_char('<'),
            },
        }
    }
}

#[derive(Clone)]
struct Map {
    tiles: Vec<Vec<Tile>>,
    guard_pos: Option<(usize, usize)>,
    visited: usize,
    loops: usize,
}

impl std::fmt::Debug for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in self.tiles.iter() {
            for tile in row.iter() {
                tile.fmt(f)?;
            }
            f.write_char('\n')?;
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Debug)]
enum State {
    Ongoing,
    Complete,
}

impl Map {
    fn new(input: &str) -> Result<Map> {
        let mut guard = None;
        let mut n_rows = None;
        let tiles = input
            .split("\n")
            .filter(|s| !s.is_empty())
            .enumerate()
            .map(|(y, s)| {
                let rows = s
                    .chars()
                    .enumerate()
                    .map(|(x, c)| {
                        Ok(match c {
                            '.' => Tile::Empty,
                            dir @ ('^' | '>' | 'v' | '<') => {
                                assert!(guard.is_none());
                                guard = Some((x, y));
                                let dir = dir.try_into()?;
                                Tile::Guard(Guard(dir), Visited::new(&dir))
                            }
                            '#' => Tile::Obstacle,
                            _ => unreachable!(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                assert_eq!(*n_rows.get_or_insert(rows.len()), rows.len());
                Ok(rows)
            })
            .collect::<Result<_>>()?;

        let visited = 1;
        let loops = 0;
        Ok(Map {
            tiles,
            guard_pos: guard,
            visited,
            loops,
        })
    }

    fn next_pos(&self) -> Option<(usize, usize)> {
        let (x, y) = self.guard_pos?;
        let Tile::Guard(guard, _) = self.tiles[y][x] else {
            return None;
        };
        let direction = guard.0;
        let (x, y) = direction.looking_at(x, y)?;
        if y < self.tiles.len() && x < self.tiles[0].len() {
            Some((x, y))
        } else {
            None
        }
    }

    fn is_in_loop(&self) -> bool {
        let Some((next_x, next_y)) = self.next_pos() else {
            return false;
        };
        let Tile::Visited(next_visited) = self.tiles[next_y][next_x] else {
            return false;
        };
        let (cur_x, cur_y) = self.guard_pos.unwrap();
        let Tile::Guard(Guard(cur_dir), _) = self.tiles[cur_y][cur_x] else {
            unreachable!()
        };
        next_visited.matches_direction(&cur_dir)
    }

    fn would_loop_here(&self) -> bool {
        let mut loop_map = self.clone();
        let Some((next_x, next_y)) = loop_map.next_pos() else {
            return false;
        };
        match loop_map.tiles[next_y][next_x] {
            Tile::Obstacle => return false,
            Tile::Visited(_) => return false,
            _ => (),
        }
        loop_map.tiles[next_y][next_x] = Tile::Obstacle;

        loop {
            match loop_map.step() {
                State::Complete => {
                    break false;
                }
                State::Ongoing => {
                    if loop_map.is_in_loop() {
                        break true;
                    }
                }
            }
        }
    }

    fn step(&mut self) -> State {
        let Some((cur_x, cur_y)) = self.guard_pos else {
            return State::Complete;
        };
        let Tile::Guard(guard, cur_visited) = self.tiles[cur_y][cur_x] else {
            unreachable!();
        };

        let next_pos = self.next_pos();
        let Some((next_x, next_y)) = next_pos else {
            let new_visited = Visited::new(&guard.0);
            self.tiles[cur_y][cur_x] = Tile::Visited(cur_visited | new_visited);
            self.guard_pos = None;
            return State::Complete;
        };

        let next_tile = self.tiles[next_y][next_x];
        match next_tile {
            Tile::Empty => {
                let new_visited = Visited::new(&guard.0);
                self.visited += 1;
                self.tiles[cur_y][cur_x] = Tile::Visited(cur_visited);
                self.tiles[next_y][next_x] = Tile::Guard(guard, new_visited);
                self.guard_pos = next_pos;
                State::Ongoing
            }
            Tile::Visited(next_visited) => {
                let new_visited = Visited::new(&guard.0);
                self.tiles[cur_y][cur_x] = Tile::Visited(cur_visited);
                self.tiles[next_y][next_x] = Tile::Guard(guard, next_visited | new_visited);
                self.guard_pos = next_pos;
                State::Ongoing
            }
            Tile::Obstacle => {
                let guard = guard.turn();
                let new_visited = Visited::new(&guard.0);
                self.tiles[cur_y][cur_x] = Tile::Guard(guard, cur_visited | new_visited);
                State::Ongoing
            }
            Tile::Guard(..) => unreachable!(),
        }
    }

    fn step_and_count_loops(&mut self) -> State {
        let next_pos = self.next_pos();

        let Some((cur_x, cur_y)) = self.guard_pos else {
            return State::Complete;
        };
        let Tile::Guard(guard, cur_visited) = self.tiles[cur_y][cur_x] else {
            unreachable!();
        };

        let Some((next_x, next_y)) = next_pos else {
            let new_visited = Visited::new(&guard.0);
            self.tiles[cur_y][cur_x] = Tile::Visited(cur_visited | new_visited);
            self.guard_pos = None;
            return State::Complete;
        };

        let next_tile = self.tiles[next_y][next_x];

        if self.would_loop_here() {
            self.loops += 1;
        }

        match next_tile {
            Tile::Empty => {
                let new_visited = Visited::new(&guard.0);
                self.visited += 1;
                self.tiles[cur_y][cur_x] = Tile::Visited(cur_visited);
                self.tiles[next_y][next_x] = Tile::Guard(guard, new_visited);
                self.guard_pos = next_pos;
                State::Ongoing
            }
            Tile::Visited(next_visited) => {
                let new_visited = Visited::new(&guard.0);
                self.tiles[cur_y][cur_x] = Tile::Visited(cur_visited);
                self.tiles[next_y][next_x] = Tile::Guard(guard, next_visited | new_visited);
                self.guard_pos = next_pos;
                State::Ongoing
            }
            Tile::Obstacle => {
                let guard = guard.turn();
                let new_visited = Visited::new(&guard.0);
                self.tiles[cur_y][cur_x] = Tile::Guard(guard, cur_visited | new_visited);
                State::Ongoing
            }
            Tile::Guard(..) => unreachable!(),
        }
    }
}

pub fn run() -> Result<()> {
    println!("day 06");
    let path = PathBuf::from("./resources/day06.txt");
    let data = util::get_data_string(&path)?;
    let mut map = Map::new(&data)?;
    let mut loop_map = map.clone();

    while let State::Ongoing = map.step() {}
    let visited = map.visited;
    println!("visited: {visited}");

    let first_step_loops = map.would_loop_here();
    while let State::Ongoing = loop_map.step_and_count_loops() {}
    let loops = match first_step_loops {
        true => loop_map.loops - 1,
        false => loop_map.loops,
    };
    println!("loops: {loops}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visited() {
        let input = "....#.....
.........#
..........
..#.......
.......#..
..........
.#..^.....
........#.
#.........
......#...";
        let mut map = Map::new(input).unwrap();
        loop {
            println!("{map:?}");
            match map.step() {
                State::Ongoing => (),
                State::Complete => break,
            }
        }
        let visited = map.visited;
        assert_eq!(visited, 41);
    }

    #[test]
    fn test_loop() {
        let input = "....#.....
.........#
..........
..#.......
.......#..
..........
.#..^.....
........#.
#.........
......#...";
        let mut map = Map::new(input).unwrap();
        loop {
            println!("{map:?}");
            match map.step_and_count_loops() {
                State::Ongoing => (),
                State::Complete => break,
            }
        }
        let loops = map.loops;
        assert_eq!(loops, 6);
    }

    #[test]
    fn test_loop_edgecase() {
        let input = "..........
....#.....
..#....#..
.#........
......#...
..........
..#.^.....
.#....#...
...#.#....
..........";
        let mut map = Map::new(input).unwrap();
        loop {
            println!("{map:?}");
            match map.step_and_count_loops() {
                State::Ongoing => (),
                State::Complete => break,
            }
        }
        let loops = map.loops;
        assert_eq!(loops, 4);
    }
}

use anyhow::Result;
use std::{collections::VecDeque, path::PathBuf};

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Pos(usize, usize);

impl Pos {
    fn parse(input: &str) -> Result<Pos> {
        use AocError::ParseError;
        let (x, y) = input.split_once(",").ok_or(ParseError)?;
        let x = x.parse()?;
        let y = y.parse()?;
        Ok(Pos(x, y))
    }
}

#[derive(Default, Copy, Clone, Debug)]
enum Byte {
    #[default]
    Empty,
    Visited,
    Corrupt,
}

struct Memory {
    size: Pos,
    bytes: Vec<Byte>,
}

impl Memory {
    fn new(size: usize) -> Memory {
        let bytes = vec![Byte::Empty; size * size];
        let size = Pos(size, size);
        Self { size, bytes }
    }

    fn at(&self, x: usize, y: usize) -> Byte {
        let Pos(width, _) = self.size;
        self.bytes[y * width + x]
    }

    fn at_mut(&mut self, x: usize, y: usize) -> &mut Byte {
        let Pos(width, _) = self.size;
        &mut self.bytes[y * width + x]
    }

    fn add_corruption(&mut self, coords: &[Pos]) {
        for &Pos(x, y) in coords {
            *self.at_mut(x, y) = Byte::Corrupt;
        }
    }

    fn reset_visited(&mut self) {
        for b in self.bytes.iter_mut() {
            if let Byte::Visited = *b {
                *b = Byte::Empty;
            }
        }
    }

    fn find_exit(&mut self) -> Option<usize> {
        struct BFSNode {
            at: Pos,
            step: usize,
        }

        self.reset_visited();
        let Pos(width, height) = self.size;
        if width == 0 || height == 0 {
            return None;
        }
        let goal = Pos(width - 1, height - 1);
        let initial = BFSNode {
            at: Pos(0, 0),
            step: 0,
        };
        let mut queue = VecDeque::new();
        queue.push_back(initial);
        while let Some(node) = queue.pop_front() {
            assert!(queue.len() < width * height);
            let BFSNode { at, step } = node;
            assert!(step <= width * height);
            let Pos(x, y) = at;

            if let Byte::Visited | Byte::Corrupt = self.at(x, y) {
                continue;
            }

            if at == goal {
                return Some(step);
            }

            *self.at_mut(x, y) = Byte::Visited;

            let step = step + 1;
            if y > 0 {
                let next = BFSNode {
                    at: Pos(x, y - 1),
                    step,
                };
                queue.push_back(next);
            }
            if x < width - 1 {
                let next = BFSNode {
                    at: Pos(x + 1, y),
                    step,
                };
                queue.push_back(next);
            }
            if y < height - 1 {
                let next = BFSNode {
                    at: Pos(x, y + 1),
                    step,
                };
                queue.push_back(next);
            }
            if x > 0 {
                let next = BFSNode {
                    at: Pos(x - 1, y),
                    step,
                };
                queue.push_back(next);
            }
        }
        None
    }

    fn find_cutoff(&mut self, corruption: &[Pos]) -> Option<Pos> {
        for &pos in corruption {
            self.add_corruption(&[pos]);
            if self.find_exit().is_none() {
                return Some(pos);
            }
        }
        None
    }
}

fn parse_corruption(input: &str) -> Result<Vec<Pos>> {
    input
        .split("\n")
        .filter(|s| !s.is_empty())
        .map(Pos::parse)
        .collect()
}

pub fn run() -> Result<()> {
    println!("day 18");
    let path = PathBuf::from("./resources/day18.txt");
    let data = util::get_data_string(&path)?;
    let corruption = parse_corruption(&data)?;

    let mut memory = Memory::new(71);
    memory.add_corruption(&corruption[..1024]);
    let steps = memory
        .find_exit()
        .ok_or(AocError::ValueError("never exit".into()))?;
    println!("Get to exit after {steps} steps");

    let cutoff = memory
        .find_cutoff(&corruption[1024..])
        .ok_or(AocError::ValueError("never cut off".into()))?;
    println!("You will get cut off by byte {cutoff:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_memory() {
        let input = "5,4
4,2
4,5
3,0
2,1
6,3
2,4
1,5
0,6
3,3
2,6
5,1
1,2
5,5
2,5
6,5
1,4
0,4
6,4
1,1
6,1
1,0
0,5
1,6
2,0";
        let corruption = parse_corruption(input).unwrap();
        let mut memory = Memory::new(7);
        memory.add_corruption(&corruption[..12]);
        let steps = memory.find_exit().unwrap();
        assert_eq!(steps, 22);
    }

    #[test]
    fn test_memory_cutoff() {
        let input = "5,4
4,2
4,5
3,0
2,1
6,3
2,4
1,5
0,6
3,3
2,6
5,1
1,2
5,5
2,5
6,5
1,4
0,4
6,4
1,1
6,1
1,0
0,5
1,6
2,0";
        let corruption = parse_corruption(input).unwrap();
        let mut memory = Memory::new(7);
        memory.add_corruption(&corruption[..12]);
        let cutoff = memory.find_cutoff(&corruption[12..]).unwrap();
        assert_eq!(cutoff, Pos(6, 1));
    }
}

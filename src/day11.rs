use anyhow::Result;
use std::{
    collections::{hash_map::Entry, HashMap},
    ops::ControlFlow,
    path::PathBuf,
};

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug)]
struct Stone(usize);

impl Stone {
    fn blink(self) -> Vec<Self> {
        match self.0 {
            0 => vec![Stone(1)],
            v if util::is_even(v.to_string().len()) => {
                let s = v.to_string();
                let mid = s.len() / 2;
                let (left, right) = s.split_at(mid);
                let left = Stone::try_from(left).expect("can split");
                let right = Stone::try_from(right).expect("can_split");
                vec![left, right]
            }
            v => vec![Stone(v * 2024)],
        }
    }
}

impl TryFrom<&str> for Stone {
    type Error = AocError;

    fn try_from(s: &str) -> std::result::Result<Self, Self::Error> {
        if s.is_empty() {
            let s = "can not make stone from empty str".to_string();
            return Err(AocError::ValueError(s));
        }
        let s = s.trim_start_matches("00");
        if s.is_empty() {
            return Ok(Stone(0));
        }

        let v = s.parse::<usize>().map_err(|_| {
            let s = format!("Could not parse to stone: {s}");
            AocError::ValueError(s)
        })?;
        Ok(Stone(v))
    }
}

fn parse_stones(input: &str) -> Result<Vec<Stone>> {
    input
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| Ok(Stone(s.parse::<usize>()?)))
        .collect()
}

fn do_blinks(mut stones: Vec<Stone>, count: usize) -> Vec<Stone> {
    for _ in 0..count {
        let test = stones.iter().map(|s| s.blink()).collect::<Vec<_>>();
        stones = test.into_iter().flatten().collect();
    }
    stones
}

#[derive(Debug)]
struct StackFrame {
    stone: Stone,
    blinks_left: usize,
}

type Cache = HashMap<(usize, usize), usize>; // map (value, blinks) -> n_stones

// find all split totals via Depth First Search
fn expand_cache(stone: Stone, count: usize, cache: &mut Cache) {
    let mut stack = Vec::with_capacity(count);

    let initial_frame = StackFrame {
        stone,
        blinks_left: count,
    };
    stack.push(initial_frame);

    while !stack.is_empty() {
        assert!(stack.len() <= count, "stack got larger than anticipated");

        let last = stack.last().expect("tested that stack not empty");
        let &StackFrame { stone, blinks_left } = last;

        // check if we already know the answer
        if let Entry::Occupied(_) = cache.entry((stone.0, blinks_left)) {
            stack.pop();
            continue;
        }
        if blinks_left == 1 {
            let value = stone.blink().len();
            cache.entry((stone.0, 1)).insert_entry(value);
            stack.pop();
            continue;
        }

        // check if we know the answer to our splits
        // if not, then we push to stack and get missing answer first
        let total_splits = stone.blink().into_iter().try_fold(0, |acc, stone| {
            let key = (stone.0, blinks_left - 1);
            match cache.entry(key) {
                Entry::Vacant(_) => ControlFlow::Break(stone),
                Entry::Occupied(e) => ControlFlow::Continue(acc + *e.get()),
            }
        });

        let total_splits = match total_splits {
            ControlFlow::Break(stone) => {
                let frame = StackFrame {
                    stone,
                    blinks_left: blinks_left - 1,
                };
                stack.push(frame);
                continue;
            }
            ControlFlow::Continue(acc) => acc,
        };

        let key = (stone.0, blinks_left);
        cache.entry(key).insert_entry(total_splits);
        stack.pop();
    }
}

fn do_blinks_cached(stones: &Vec<Stone>, count: usize) -> usize {
    let mut cache: Cache = HashMap::new(); // map (value, blinks) -> n_stones
    for &stone in stones {
        expand_cache(stone, count, &mut cache);
    }

    let total = stones
        .iter()
        .map(|s| {
            let key = (s.0, count);
            let entry = cache.get(&key).expect("has been filled in expand_cache");
            entry
        })
        .sum();

    total
}

pub fn run() -> Result<()> {
    println!("day 11");
    let path = PathBuf::from("./resources/day11.txt");
    let data = util::get_data_string(&path)?;
    let stones_orig = parse_stones(&data)?;
    let stones = do_blinks(stones_orig.clone(), 25);
    println!("after 25 blinks we have: {} stones", stones.len());
    let total = do_blinks_cached(&stones_orig, 75);
    println!("after 75 blinks we have: {total} stones");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stone_from_str() {
        let s = "0";
        let stone = Stone::try_from(s).unwrap();
        assert_eq!(stone.0, 0);
        let s = "000";
        let stone = Stone::try_from(s).unwrap();
        assert_eq!(stone.0, 0);
        let s = "00000001";
        let stone = Stone::try_from(s).unwrap();
        assert_eq!(stone.0, 1);
    }

    #[test]
    fn test_single_blink() {
        let input = "0 1 125 17 2002";
        let stones = parse_stones(input).unwrap();
        assert_eq!(stones.len(), 5);
        let stones = do_blinks(stones, 1);
        assert_eq!(stones.len(), 7);
    }

    #[test]
    fn test_two_blinks() {
        let input = "0 1 125 17 2002";
        let stones = parse_stones(input).unwrap();
        assert_eq!(stones.len(), 5);
        let stones = do_blinks(stones, 2);
        assert_eq!(stones.len(), 10);
    }

    #[test]
    fn test_example() {
        let input = "125 17";
        let stones = parse_stones(input).unwrap();
        assert_eq!(stones.len(), 2);
        let stones = do_blinks(stones, 6);
        assert_eq!(stones.len(), 22);
        let stones = do_blinks(stones, 25 - 6);
        assert_eq!(stones.len(), 55312);
    }

    #[test]
    fn test_example_cached() {
        let input = "125 17";
        let stones = parse_stones(input).unwrap();
        assert_eq!(stones.len(), 2);
        let total = do_blinks_cached(&stones, 25);
        assert_eq!(total, 55312);
    }
}

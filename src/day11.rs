use anyhow::Result;
use std::path::PathBuf;

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug)]
struct Stone(usize);

impl Stone {
    fn blink(self) -> Result<Vec<Self>> {
        match self.0 {
            0 => Ok(vec![Stone(1)]),
            v if util::is_even(v.to_string().len()) => {
                let s = v.to_string();
                let mid = s.len() / 2;
                let (left, right) = s.split_at(mid);
                let left = Stone::try_from(left)?;
                let right = Stone::try_from(right)?;
                Ok(vec![left, right])
            }
            v => Ok(vec![Stone(v * 2024)]),
        }
    }
}

impl TryFrom<&str> for Stone {
    type Error = AocError;

    fn try_from(s: &str) -> std::result::Result<Self, Self::Error> {
        if s.len() == 0 {
            let s = format!("can not make stone from: {s}");
            return Err(AocError::ValueError(s));
        }
        let s = s.trim_start_matches("00");
        if s.len() == 0 {
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
        .filter(|s| s.len() != 0)
        .map(|s| Ok(Stone(s.parse::<usize>()?)))
        .collect()
}

fn do_blinks(mut stones: Vec<Stone>, count: usize) -> Result<Vec<Stone>> {
    for _ in 0..count {
        let test = stones
            .iter()
            .map(|s| s.blink())
            .collect::<Result<Vec<_>>>()?;
        stones = test.into_iter().flatten().collect();
    }
    Ok(stones)
}

pub fn run() -> Result<()> {
    println!("day 11");
    let path = PathBuf::from("./resources/day11.txt");
    let data = util::get_data_string(&path)?;
    let stones = parse_stones(&data)?;
    let stones = do_blinks(stones, 25)?;
    println!("after 25 blinks we have: {} stones", stones.len());
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
        let stones = do_blinks(stones, 1).unwrap();
        assert_eq!(stones.len(), 7);
    }

    #[test]
    fn test_two_blinks() {
        let input = "0 1 125 17 2002";
        let stones = parse_stones(input).unwrap();
        assert_eq!(stones.len(), 5);
        let stones = do_blinks(stones, 2).unwrap();
        assert_eq!(stones.len(), 10);
    }

    #[test]
    fn test_example() {
        let input = "125 17";
        let stones = parse_stones(input).unwrap();
        assert_eq!(stones.len(), 2);
        let stones = do_blinks(stones, 6).unwrap();
        assert_eq!(stones.len(), 22);
        let stones = do_blinks(stones, 25 - 6).unwrap();
        assert_eq!(stones.len(), 55312);
    }
}

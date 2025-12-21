use anyhow::Result;
use std::{iter::zip, path::PathBuf};

use crate::util::{self};

struct Lock {
    heights: [usize; 5],
}

struct Key {
    heights: [usize; 5],
}

fn parse_locks_and_keys(input: &str) -> Result<(Vec<Lock>, Vec<Key>)> {
    let mut locks = Vec::new();
    let mut keys = Vec::new();
    for schematic in input.split("\n\n") {
        let parser = |_x, _y, c| Ok(c);
        let map = util::parse_tiles(schematic, parser)?;

        let cols = map[0].len();
        let rows = map.len();
        assert_eq!(rows, 7);
        assert_eq!(cols, 5);

        let mut heights = [0; 5];

        for (x, row) in map.iter().enumerate() {
            for c in row.iter() {
                if *c == '#' {
                    heights[x] += 1;
                }
            }
        }
        if map[0][0] == '.' {
            let key = Key { heights };
            keys.push(key);
        } else {
            let lock = Lock { heights };
            locks.push(lock);
        }
    }
    Ok((locks, keys))
}

fn count_overlap_free_pairings(locks: &[Lock], keys: &[Key]) -> usize {
    let mut count = 0;

    for lock in locks {
        for key in keys {
            let fitting_cols = zip(lock.heights, key.heights)
                .filter(|(l, k)| l + k <= 5)
                .count();
            if fitting_cols == 5 {
                count += 1;
            }
        }
    }
    count
}

pub fn run() -> Result<()> {
    println!("day 25");
    let path = PathBuf::from("./resources/day25.txt");
    let data = util::get_data_string(&path)?;
    let (locks, keys) = parse_locks_and_keys(&data).unwrap();
    let pairings = count_overlap_free_pairings(&locks, &keys);
    println!("got {pairings} many key-lock pairings");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_wires() {
        let input = "#####
.####
.####
.####
.#.#.
.#...
.....

#####
##.##
.#.##
...##
...#.
...#.
.....

.....
#....
#....
#...#
#.#.#
#.###
#####

.....
.....
#.#..
###..
###.#
###.#
#####

.....
.....
.....
#....
#.#..
#.#.#
#####";
        let (locks, keys) = parse_locks_and_keys(input).unwrap();
        let pairings = count_overlap_free_pairings(&locks, &keys);
        assert_eq!(pairings, 3);
    }
}

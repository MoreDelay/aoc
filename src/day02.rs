use anyhow::Result;
use std::{
    cmp::{max, min},
    ops::BitAnd,
    path::PathBuf,
};

use crate::util;

fn validate_record_dampened(row: &[usize], skips: usize) -> bool {
    let skips = min(row.len() - 1, skips);

    // dynamic program: resolve recursive evaluation from the back
    // mark if record valid using 0, 1, ... skips
    let mut increasing_table = vec![vec![false; row.len()]; skips + 1];
    let mut decreasing_table = vec![vec![false; row.len()]; skips + 1];

    let gradual = |a, b| (max(a, b) - min(a, b)) <= 3;
    let increasing = |a, b| (a < b) && gradual(a, b);
    let decreasing = |a, b| (a > b) && gradual(a, b);

    // initialize: at the end you can skip to completion
    for skip in 0..=skips {
        for index in 0..=skip {
            increasing_table[skip][row.len() - 1 - index] = true;
            decreasing_table[skip][row.len() - 1 - index] = true;
        }
    }

    for skip in 0..=skips {
        for index in (0..row.len() - 1 - skip).rev() {
            let cur = row[index];
            for used_skips in 0..=skip {
                let next_index = index + used_skips + 1;
                let next = row[next_index];

                let incr_valid =
                    increasing_table[skip - used_skips][next_index].bitand(increasing(cur, next));
                let decr_valid =
                    decreasing_table[skip - used_skips][next_index].bitand(decreasing(cur, next));

                increasing_table[skip][index] |= incr_valid;
                decreasing_table[skip][index] |= decr_valid;
            }
        }

        // first entry can be skipped entirely
        if skip > 0 {
            increasing_table[skip][0] |= increasing_table[skip - 1][1];
            decreasing_table[skip][0] |= decreasing_table[skip - 1][1];
        }
    }

    increasing_table[skips][0] || decreasing_table[skips][0]
}
pub fn run() -> Result<()> {
    println!("day 02");
    let path = PathBuf::from("./resources/day02.txt");
    let data = util::get_data_rows(&path)?;

    let increasing = |a, b| a < b;
    let decreasing = |a, b| a > b;
    let gradual = |a, b| (max(a, b) - min(a, b)) <= 3;

    let valid_records = data
        .iter()
        .filter(|row| {
            row.windows(2)
                .flat_map(<&[usize; 2]>::try_from)
                .all(|&[a, b]| increasing(a, b) && gradual(a, b))
                || row
                    .windows(2)
                    .flat_map(<&[usize; 2]>::try_from)
                    .all(|&[a, b]| decreasing(a, b) && gradual(a, b))
        })
        .count();
    println!("valid={valid_records}");

    let dampened_records: Vec<_> = data
        .iter()
        .filter(|&row| validate_record_dampened(row, 1))
        .collect();
    let dampened_valid = dampened_records.len();
    println!("dampened={dampened_valid}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one() {
        let row = vec![1; 1];
        let valid = validate_record_dampened(&row, 1);
        assert!(valid);
    }

    #[test]
    fn test_small() {
        let row = vec![50, 51];
        let valid = validate_record_dampened(&row, 1);
        assert!(valid);
    }

    #[test]
    fn test_incr() {
        let row = vec![50, 51, 52, 53, 54];
        let valid = validate_record_dampened(&row, 1);
        assert!(valid);
    }

    #[test]
    fn test_incr_skip_first() {
        let row = vec![52, 51, 52, 53, 54];
        let valid = validate_record_dampened(&row, 1);
        assert!(valid);
    }

    #[test]
    fn test_incr_skip_last() {
        let row = vec![50, 51, 52, 53, 52];
        let valid = validate_record_dampened(&row, 1);
        assert!(valid);
    }

    #[test]
    fn test_incr_skip_middle() {
        let row = vec![50, 51, 46, 53, 54];
        let valid = validate_record_dampened(&row, 1);
        assert!(valid);
    }

    #[test]
    fn test_incr_skip_twice() {
        let row = vec![50, 51, 51, 53, 53];
        let valid = validate_record_dampened(&row, 1);
        assert!(!valid);
    }

    #[test]
    fn test_decr() {
        let row = vec![66, 63, 59, 58, 56];
        let valid = validate_record_dampened(&row, 1);
        assert!(!valid);
    }

    #[test]
    fn test_decr_skip_first() {
        let row = vec![68, 61, 60, 59, 56];
        let valid = validate_record_dampened(&row, 1);
        assert!(valid);
    }

    #[test]
    fn test_decr_skip_last() {
        let row = vec![62, 61, 60, 59, 52];
        let valid = validate_record_dampened(&row, 1);
        assert!(valid);
    }

    #[test]
    fn test_decr_skip_middle() {
        let row = vec![38, 34, 37, 35, 34];
        let valid = validate_record_dampened(&row, 1);
        assert!(valid);
    }

    #[test]
    fn test_decr_skip_twice() {
        let row = vec![62, 62, 60, 60, 59];
        let valid = validate_record_dampened(&row, 1);
        assert!(!valid);
    }
}

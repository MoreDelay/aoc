use anyhow::Result;
use std::{
    cmp::{max, min},
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    iter::zip,
    path::{Path, PathBuf},
};

fn get_data_fixed_columns<const C: usize>(path: &Path) -> Result<[Vec<usize>; C]> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut result: [Vec<usize>; C] = vec![Vec::new(); C].try_into().unwrap();
    for line in reader.lines() {
        let parsed: Vec<Result<_>> = line?
            .split_whitespace()
            .map(|val| val.parse::<usize>().map_err(|e| e.into()))
            .collect();
        let parsed: Result<Vec<usize>> = parsed.into_iter().collect();
        let parsed = parsed?;
        assert_eq!(parsed.len(), C);
        for index in 0..C {
            result[index].push(parsed[index]);
        }
    }
    Ok(result)
}

fn get_data_rows(path: &Path) -> Result<Vec<Vec<usize>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut result = Vec::new();

    for line in reader.lines() {
        let parsed: Vec<Result<_>> = line?
            .split_whitespace()
            .map(|val| val.parse::<usize>().map_err(|e| e.into()))
            .collect();
        let parsed: Result<Vec<usize>> = parsed.into_iter().collect();
        let parsed = parsed?;
        result.push(parsed);
    }
    Ok(result)
}

fn day_01() -> Result<()> {
    println!("day 01");
    let path = PathBuf::from("./resources/day01.txt");
    let [mut left, mut right] = get_data_fixed_columns(&path)?;

    left.sort();
    right.sort();

    let distance: usize = zip(&mut *left, &mut *right)
        .into_iter()
        .map(|(a, b)| a.abs_diff(*b))
        .sum();
    println!("distance={distance}");

    let mut right_freq = HashMap::new();
    for val in right.iter() {
        *right_freq.entry(val).or_insert(0usize) += 1;
    }
    let similarity: usize = left
        .iter()
        .map(|val| val * *right_freq.entry(val).or_default())
        .sum();
    println!("similarity={similarity}");
    Ok(())
}

fn brute_force(row: &[usize]) -> bool {
    let gradual = |a, b| (max(a, b) - min(a, b)) <= 3;
    let increasing = |a, b| (a < b) && gradual(a, b);
    let decreasing = |a, b| (a > b) && gradual(a, b);

    let incr = row
        .windows(2)
        .flat_map(<&[usize; 2]>::try_from)
        .filter(|&&[a, b]| !increasing(a, b))
        .count()
        == 0;
    let decr = row
        .windows(2)
        .flat_map(<&[usize; 2]>::try_from)
        .filter(|&&[a, b]| !decreasing(a, b))
        .count()
        == 0;
    if incr || decr {
        return true;
    }

    for i in 0..row.len() {
        let mut temp = Vec::with_capacity(row.len() - 1);
        for left in 0..i {
            temp.push(row[left]);
        }
        for right in i + 1..row.len() {
            temp.push(row[right]);
        }
        let incr = temp
            .windows(2)
            .flat_map(<&[usize; 2]>::try_from)
            .filter(|&&[a, b]| !increasing(a, b))
            .count()
            == 0;
        let decr = temp
            .windows(2)
            .flat_map(<&[usize; 2]>::try_from)
            .filter(|&&[a, b]| !decreasing(a, b))
            .count()
            == 0;
        if incr || decr {
            return true;
        }
    }
    return false;
}

fn validate_record_dampened(row: &[usize], skips: usize) -> bool {
    let skips = min(row.len() - 1, skips);

    // dynamic program:
    // mark if record valid using 0, 1, ... skips
    // value indicates the index of last value to compare to
    let mut increasing_table = vec![vec![None; row.len()]; skips + 1];
    let mut decreasing_table = vec![vec![None; row.len()]; skips + 1];

    let gradual = |a, b| (max(a, b) - min(a, b)) <= 3;
    let increasing = |a, b| (a < b) && gradual(a, b);
    let decreasing = |a, b| (a > b) && gradual(a, b);

    // initialize: skipping all values until no longer able to is valid
    for index in 0..=skips {
        increasing_table[index][index] = Some(index);
        decreasing_table[index][index] = Some(index);
    }

    for index in 1..row.len() {
        let cur = row[index];

        let no_skip_incr = increasing_table[0][index - 1]
            .is_some_and(|last_index| increasing(row[last_index], cur))
            .then_some(index);
        let no_skip_decr = decreasing_table[0][index - 1]
            .is_some_and(|last_index| decreasing(row[last_index], cur))
            .then_some(index);

        increasing_table[0][index] = no_skip_incr;
        decreasing_table[0][index] = no_skip_decr;
    }

    for skip in 1..=skips {
        for index in (skip + 1)..row.len() {
            let cur = row[index];

            let no_skip_incr = increasing_table[skip][index - 1]
                .is_some_and(|last_index| increasing(row[last_index], cur))
                .then_some(index);
            let no_skip_decr = decreasing_table[skip][index - 1]
                .is_some_and(|last_index| decreasing(row[last_index], cur))
                .then_some(index);

            let skip_here_incr = increasing_table[skip - 1][index - 1];
            let skip_here_decr = decreasing_table[skip - 1][index - 1];

            increasing_table[skip][index] = skip_here_incr.or(no_skip_incr);
            decreasing_table[skip][index] = skip_here_decr.or(no_skip_decr);
        }
    }

    increasing_table[skips][row.len() - 1].is_some()
        || decreasing_table[skips][row.len() - 1].is_some()
}

fn day_02() -> Result<()> {
    println!("day 02");
    let path = PathBuf::from("./resources/day02.txt");
    let data = get_data_rows(&path)?;

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

    let dampened_records: Vec<_> = data.iter().filter(|&row| brute_force(row)).collect();
    let dampened_valid = dampened_records.len();

    println!("dampened={dampened_valid}");
    Ok(())
}

fn main() -> Result<()> {
    day_01()?;
    day_02()?;
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
        let row = vec![62, 61, 60, 61, 59];
        let valid = validate_record_dampened(&row, 1);
        assert!(valid);
    }

    #[test]
    fn test_decr_skip_twice() {
        let row = vec![62, 62, 60, 60, 59];
        let valid = validate_record_dampened(&row, 1);
        assert!(!valid);
    }

    #[test]
    fn test_example_1() {
        let row = vec![7, 6, 4, 2, 1];
        let valid = validate_record_dampened(&row, 1);
        assert!(valid);
    }
}

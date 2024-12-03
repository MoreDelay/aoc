use anyhow::Result;
use std::{
    cmp::{max, min},
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read},
    iter::zip,
    ops::BitAnd,
    path::{Path, PathBuf},
};

fn get_data_string(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut result = String::new();
    reader.read_to_string(&mut result)?;
    Ok(result)
}

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

    let dampened_records: Vec<_> = data
        .iter()
        .filter(|&row| validate_record_dampened(row, 1))
        .collect();
    let dampened_valid = dampened_records.len();
    println!("dampened={dampened_valid}");

    Ok(())
}

struct Mul(usize, usize);

fn parse_mul(input: &str) -> Vec<Mul> {
    let mut result = Vec::new();
    let mut remaining = &input[..];
    const MAX_PATTERN: usize = 12;

    fn parse_val(s: &str) -> Option<usize> {
        if s.len() > 3 {
            None
        } else {
            s.parse().ok()
        }
    }

    while remaining.len() > 0 {
        let index = match remaining.find("mul(") {
            Some(index) => index,
            None => break,
        };
        remaining = &remaining[index..];
        let slice_end = min(MAX_PATTERN, remaining.len());
        let check = &remaining[..slice_end];
        let comma = match check.find(",") {
            Some(index) => index,
            None => {
                remaining = &remaining[4..];
                continue;
            }
        };
        let first_val = match parse_val(&check[4..comma]) {
            Some(val) => val,
            None => {
                remaining = &remaining[4..];
                continue;
            }
        };
        let closing = match check.find(")") {
            Some(index) => index,
            None => {
                remaining = &remaining[4..];
                continue;
            }
        };
        let second_val = match parse_val(&check[comma + 1..closing]) {
            Some(val) => val,
            None => {
                remaining = &remaining[4..];
                continue;
            }
        };
        result.push(Mul(first_val, second_val));
        remaining = &remaining[4..];
    }

    result
}

fn parse_mul_conditional(input: &str) -> Vec<Mul> {
    let mut result = Vec::new();
    let mut remaining = &input[..];

    while remaining.len() > 0 {
        // we are always enabled at the beginning of a loop
        let Some(dont_index) = remaining.find("don't()") else {
            result.extend(parse_mul(remaining));
            break;
        };
        let mul_slice = &remaining[..dont_index];
        result.extend(parse_mul(mul_slice));

        // skip forward until we are enabled again
        let searching_do = &remaining[dont_index + 7..];
        let Some(do_index) = searching_do.find("do()") else {
            break;
        };
        remaining = &remaining[dont_index + 7 + do_index + 4..];
    }

    result
}

fn day_03() -> Result<()> {
    println!("day 03");
    let path = PathBuf::from("./resources/day03.txt");
    let data = get_data_string(&path)?;
    let muls = parse_mul(&data);

    let result = muls
        .iter()
        .map(|&Mul(a, b)| a * b)
        .reduce(|a, b| a + b)
        .unwrap_or(0);
    println!("result={result}");

    let cond_muls = parse_mul_conditional(&data);

    let cond_result = cond_muls
        .iter()
        .map(|&Mul(a, b)| a * b)
        .reduce(|a, b| a + b)
        .unwrap_or(0);
    println!("cond_result={cond_result}");

    Ok(())
}

fn main() -> Result<()> {
    day_01()?;
    day_02()?;
    day_03()?;
    Ok(())
}

#[cfg(test)]
mod test_day_03 {
    use super::*;

    #[test]
    fn test_parse() {
        let s = "a mul(1,2) to mul(43,654)";
        let muls = parse_mul(s);
        assert_eq!(muls.len(), 2);
    }

    #[test]
    fn test_parse_cond() {
        let s = "a don't() mul(1,2) to do() mul(43,654)";
        let muls = parse_mul_conditional(s);
        assert_eq!(muls.len(), 1);
    }
}

#[cfg(test)]
mod test_day_02 {
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

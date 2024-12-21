use anyhow::Result;
use std::{cmp::min, path::PathBuf};

use crate::util;

struct Mul(usize, usize);

fn parse_mul(input: &str) -> Vec<Mul> {
    let mut result = Vec::new();
    let mut remaining = input;
    const MAX_PATTERN: usize = 12;

    fn parse_val(s: &str) -> Option<usize> {
        if s.len() > 3 {
            None
        } else {
            s.parse().ok()
        }
    }

    while !remaining.is_empty() {
        let Some(start_index) = remaining.find("mul(") else {
            break;
        };
        // skip forward for next iteration, but still check current mul
        let check_slice = &remaining[start_index..];
        remaining = &remaining[start_index + 4..];

        let slice_end = min(MAX_PATTERN, check_slice.len());
        let check_slice = &check_slice[..slice_end];

        let Some(comma) = check_slice.find(",") else {
            continue;
        };
        let Some(first_val) = parse_val(&check_slice[4..comma]) else {
            continue;
        };

        let check_slice = &check_slice[comma + 1..];

        let Some(closing) = check_slice.find(")") else {
            continue;
        };
        let Some(second_val) = parse_val(&check_slice[..closing]) else {
            continue;
        };
        result.push(Mul(first_val, second_val));
    }

    result
}

fn parse_mul_conditional(input: &str) -> Vec<Mul> {
    let mut result = Vec::new();
    let mut remaining = input;

    while !remaining.is_empty() {
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

pub fn run() -> Result<()> {
    println!("day 03");
    let path = PathBuf::from("./resources/day03.txt");
    let data = util::get_data_string(&path)?;
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

#[cfg(test)]
mod tests {
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

use anyhow::Result;
use std::path::PathBuf;

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Stripe {
    White,
    Blue,
    Black,
    Red,
    Green,
}

impl Stripe {
    fn parse(c: char) -> Result<Stripe> {
        use Stripe::*;
        match c {
            'w' => Ok(White),
            'u' => Ok(Blue),
            'b' => Ok(Black),
            'r' => Ok(Red),
            'g' => Ok(Green),
            _ => Err(AocError::ParseError.into()),
        }
    }
}

#[derive(Clone, Debug)]
struct Towel {
    stripes: Vec<Stripe>,
}

#[derive(Clone, Debug)]
struct Pattern {
    stripes: Vec<Stripe>,
}

impl Pattern {
    fn can_combine(&self, towels: &[Towel]) -> usize {
        let n_stripes = self.stripes.len();
        let mut table = vec![0; n_stripes + 1];
        table[n_stripes] = 1;
        for split in (0..n_stripes).rev() {
            let (_, to_solve) = self.stripes.split_at(split);
            for towel in towels {
                let prefix = towel.stripes.as_slice();
                let Some(rest) = to_solve.strip_prefix(prefix) else {
                    continue;
                };
                let rest = rest.len();
                let table_index = n_stripes - rest;
                table[split] += table[table_index];
            }
        }
        table[0]
    }
}

fn parse_stripes(input: &str) -> Result<(Vec<Towel>, Vec<Pattern>)> {
    use AocError::ParseError;

    let (towels, patterns) = input.split_once("\n").ok_or(ParseError)?;
    let towels = towels
        .split(", ")
        .filter(|s| !s.is_empty())
        .map(|s| {
            let stripes = s.chars().map(Stripe::parse).collect::<Result<Vec<_>>>()?;
            Ok(Towel { stripes })
        })
        .collect::<Result<Vec<_>>>()?;

    let patterns = patterns
        .split("\n")
        .filter(|s| !s.is_empty())
        .map(|s| {
            let stripes = s.chars().map(Stripe::parse).collect::<Result<Vec<_>>>()?;
            Ok(Pattern { stripes })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok((towels, patterns))
}

fn count_options(patterns: &[Pattern], towels: &[Towel]) -> Vec<usize> {
    patterns.iter().map(|p| p.can_combine(&towels)).collect()
}

pub fn run() -> Result<()> {
    println!("day 19");
    let path = PathBuf::from("./resources/day19.txt");
    let data = util::get_data_string(&path)?;
    let (towels, patterns) = parse_stripes(&data).unwrap();
    let options = count_options(&patterns, &towels);
    let possible = options.iter().filter(|&&v| v > 0).count();
    println!("can create {possible} patterns");

    let arrangements: usize = options.iter().sum();
    println!("we have {arrangements} options to create patterns");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_patterns() {
        let input = "r, wr, b, g, bwu, rb, gb, br

brwrr
bggr
gbbr
rrbgbr
ubwu
bwurrg
brgr
bbrgwb";
        let (towels, patterns) = parse_stripes(input).unwrap();
        let options = count_options(&patterns, &towels);
        let possible = options.iter().filter(|&&v| v > 0).count();
        assert_eq!(possible, 6);
    }

    #[test]
    fn test_count_pattern_arrangements() {
        let input = "r, wr, b, g, bwu, rb, gb, br

brwrr
bggr
gbbr
rrbgbr
ubwu
bwurrg
brgr
bbrgwb";
        let (towels, patterns) = parse_stripes(input).unwrap();
        let options = count_options(&patterns, &towels);
        let arrangements: usize = options.iter().sum();
        assert_eq!(arrangements, 16);
    }
}

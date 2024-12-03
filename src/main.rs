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
    Ok(())
}

fn main() -> Result<()> {
    day_01()?;
    day_02()?;
    Ok(())
}

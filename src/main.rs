use anyhow::Result;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    iter::zip,
    path::{Path, PathBuf},
};

fn get_data(path: &Path) -> Result<(Vec<usize>, Vec<usize>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let result = reader.lines().map(|line| -> Result<(usize, usize)> {
        let parsed: Vec<Result<_>> = line?
            .split_whitespace()
            .map(|val| val.parse::<usize>().map_err(|e| e.into()))
            .collect();
        let parsed: Result<Vec<usize>> = parsed.into_iter().collect();
        let parsed = parsed?;
        assert_eq!(parsed.len(), 2);
        Ok((parsed[0], parsed[1]))
    });

    result.collect()
}

fn day_01() -> Result<()> {
    println!("day 01");
    let path = PathBuf::from("./resources/data.txt");
    let (mut left, mut right) = get_data(&path)?;

    left.sort();
    right.sort();

    let distance: usize = zip(&left, &right)
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

fn main() -> Result<()> {
    day_01()?;
    Ok(())
}

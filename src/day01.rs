use anyhow::Result;
use std::{collections::HashMap, iter::zip, path::PathBuf};

use crate::util;

pub fn run() -> Result<()> {
    println!("day 01");
    let path = PathBuf::from("./resources/day01.txt");
    let [mut left, mut right] = util::get_data_fixed_columns(&path)?;

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

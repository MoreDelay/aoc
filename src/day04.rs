use anyhow::Result;
use std::path::PathBuf;

use crate::util;

fn mapping_fn(c: char, mapping: &str) -> u8 {
    mapping
        .find(c)
        .map(|i| i as u8)
        .unwrap_or(mapping.len() as u8)
}

fn convert_to_vec_of_vecs(input: &str, mapping: &str) -> Vec<Vec<u8>> {
    let res = input
        .split("\n")
        .filter(|s| s.len() > 0)
        .map(|s| {
            s.chars()
                .map(|c| mapping_fn(c, mapping))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // make sure each inner vector has the same length
    if res.len() > 0 {
        let length = res[0].len();
        res.iter().skip(1).for_each(|v| assert_eq!(v.len(), length));
    }
    res
}

fn cmp_fwd_bwd(mut container: Vec<u8>, pattern: &[u8]) -> usize {
    let mut total = 0;
    total += (&container == pattern) as usize;
    container.reverse();
    total += (&container == pattern) as usize;
    total
}

fn cmp_x_diagonals(mut diag1: Vec<u8>, mut diag2: Vec<u8>, pattern: &[u8]) -> usize {
    assert_eq!(diag1.len(), diag2.len());

    fn rotate(diag1: &mut Vec<u8>, diag2: &mut Vec<u8>) {
        let last = diag2.len() - 1;
        std::mem::swap(&mut diag2[last], &mut diag1[0]);
        std::mem::swap(&mut diag1[0], &mut diag2[0]);
        std::mem::swap(&mut diag2[0], &mut diag1[last]);
    }

    let mut total = 0;
    total += (diag1 == pattern && diag2 == pattern) as usize;
    rotate(&mut diag1, &mut diag2);
    total += (diag1 == pattern && diag2 == pattern) as usize;
    rotate(&mut diag1, &mut diag2);
    total += (diag1 == pattern && diag2 == pattern) as usize;
    rotate(&mut diag1, &mut diag2);
    total += (diag1 == pattern && diag2 == pattern) as usize;
    total
}

fn find_pattern_1d(x: usize, y: usize, data: &Vec<Vec<u8>>, pattern: &[u8]) -> usize {
    let mut total = 0;

    // horizontal
    if x + pattern.len() <= data[0].len() {
        let horizontal = data[y][x..x + pattern.len()].to_vec();
        total += cmp_fwd_bwd(horizontal, pattern);
    }
    //vertical
    if y + pattern.len() <= data.len() {
        let mut vertical = Vec::with_capacity(pattern.len());
        for i in 0..pattern.len() {
            vertical.push(data[y + i][x]);
        }
        total += cmp_fwd_bwd(vertical, pattern);
    }

    if x + pattern.len() <= data[0].len() && y + pattern.len() <= data.len() {
        let mut diagonal_1 = Vec::with_capacity(pattern.len());
        let mut diagonal_2 = Vec::with_capacity(pattern.len());
        for i in 0..pattern.len() {
            diagonal_1.push(data[y + i][x + i]);
            diagonal_2.push(data[y + i][x + pattern.len() - 1 - i]);
        }
        total += cmp_fwd_bwd(diagonal_1, pattern);
        total += cmp_fwd_bwd(diagonal_2, pattern);
    }
    total
}

fn find_pattern_2d(x: usize, y: usize, data: &Vec<Vec<u8>>, pattern: &[u8]) -> usize {
    let mut total = 0;

    if x + pattern.len() <= data[0].len() && y + pattern.len() <= data.len() {
        let mut diagonal_1 = Vec::with_capacity(pattern.len());
        let mut diagonal_2 = Vec::with_capacity(pattern.len());
        for i in 0..pattern.len() {
            diagonal_1.push(data[y + i][x + i]);
            diagonal_2.push(data[y + i][x + pattern.len() - 1 - i]);
        }
        total += cmp_x_diagonals(diagonal_1, diagonal_2, pattern);
    }
    total
}

fn find_patterns<'a, F>(data: &'a Vec<Vec<u8>>, find_pattern: F) -> usize
where
    F: Fn(usize, usize, &'a Vec<Vec<u8>>) -> usize,
{
    let mut total = 0;
    for y in 0..data.len() {
        for x in 0..data[0].len() {
            total += find_pattern(x, y, data);
        }
    }
    total
}

pub fn run() -> Result<()> {
    println!("day 04");
    let path = PathBuf::from("./resources/day04.txt");
    let data = util::get_data_string(&path)?;
    let pattern = "XMAS";
    let data = convert_to_vec_of_vecs(&data, pattern);
    let pattern = pattern
        .chars()
        .map(|c| mapping_fn(c, pattern))
        .collect::<Vec<_>>();
    let find_pattern = |x, y, v| find_pattern_1d(x, y, v, &pattern);
    let count = find_patterns(&data, find_pattern);
    println!("pattern count 1d: {count}");
    let pattern = &pattern[1..];
    let find_pattern = |x, y, v| find_pattern_2d(x, y, v, pattern);
    let count = find_patterns(&data, find_pattern);
    println!("pattern count 2d: {count}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_patterns_horizontal() {
        let pattern: Vec<u8> = vec![0, 1, 2, 3];
        let data: Vec<Vec<u8>> = vec![vec![4, 4, 0, 1, 2, 3, 4]];
        let find_pattern = |x, y, v| find_pattern_1d(x, y, v, &pattern);
        let total = find_patterns(&data, find_pattern);
        assert_eq!(total, 1);
    }

    #[test]
    fn test_find_patterns_vertical() {
        let pattern: Vec<u8> = vec![0, 1, 2, 3];
        let data: Vec<Vec<u8>> = vec![
            vec![4],
            vec![4],
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4],
        ];
        let find_pattern = |x, y, v| find_pattern_1d(x, y, v, &pattern);
        let total = find_patterns(&data, find_pattern);
        assert_eq!(total, 1);
    }

    #[test]
    fn test_find_patterns_diagonal_1() {
        let pattern: Vec<u8> = vec![0, 1, 2, 3];
        let data: Vec<Vec<u8>> = vec![
            vec![3, 4, 4, 4, 4, 4],
            vec![4, 2, 4, 4, 4, 4],
            vec![4, 4, 1, 4, 4, 4],
            vec![4, 4, 4, 0, 4, 4],
            vec![4, 4, 4, 4, 4, 4],
            vec![4, 4, 4, 4, 4, 4],
        ];
        let find_pattern = |x, y, v| find_pattern_1d(x, y, v, &pattern);
        let total = find_patterns(&data, find_pattern);
        assert_eq!(total, 1);
    }

    #[test]
    fn test_find_patterns_diagonal_2() {
        let pattern: Vec<u8> = vec![0, 1, 2, 3];
        let data: Vec<Vec<u8>> = vec![
            vec![4, 4, 4, 4, 4, 4],
            vec![4, 4, 4, 4, 3, 3],
            vec![4, 4, 4, 2, 2, 4],
            vec![4, 4, 1, 1, 4, 4],
            vec![4, 0, 0, 4, 4, 4],
            vec![4, 4, 4, 4, 4, 4],
        ];
        let find_pattern = |x, y, v| find_pattern_1d(x, y, v, &pattern);
        let total = find_patterns(&data, find_pattern);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_find_patterns_multiple() {
        let pattern: Vec<u8> = vec![0, 1, 2, 3];
        let data: Vec<Vec<u8>> = vec![
            vec![4, 0, 4, 4, 4, 4],
            vec![3, 1, 4, 3, 3, 4],
            vec![4, 2, 3, 2, 1, 0],
            vec![4, 3, 1, 1, 4, 4],
            vec![4, 0, 4, 0, 4, 4],
            vec![4, 4, 3, 2, 1, 0],
        ];
        let find_pattern = |x, y, v| find_pattern_1d(x, y, v, &pattern);
        let total = find_patterns(&data, find_pattern);
        assert_eq!(total, 6);
    }
}

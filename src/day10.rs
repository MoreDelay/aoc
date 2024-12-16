use anyhow::Result;
use std::{collections::HashSet, path::PathBuf};

use crate::util::{self, AocError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tile {
    H0,
    H1,
    H2,
    H3,
    H4,
    H5,
    H6,
    H7,
    H8,
    H9,
}

impl TryFrom<char> for Tile {
    type Error = AocError;

    fn try_from(value: char) -> std::result::Result<Self, Self::Error> {
        match value {
            '0' => Ok(Tile::H0),
            '1' => Ok(Tile::H1),
            '2' => Ok(Tile::H2),
            '3' => Ok(Tile::H3),
            '4' => Ok(Tile::H4),
            '5' => Ok(Tile::H5),
            '6' => Ok(Tile::H6),
            '7' => Ok(Tile::H7),
            '8' => Ok(Tile::H8),
            '9' => Ok(Tile::H9),
            _ => Err(AocError::ParseError),
        }
    }
}

impl Tile {
    fn next(&self) -> Result<Tile, AocError> {
        match self {
            Tile::H0 => Ok(Tile::H1),
            Tile::H1 => Ok(Tile::H2),
            Tile::H2 => Ok(Tile::H3),
            Tile::H3 => Ok(Tile::H4),
            Tile::H4 => Ok(Tile::H5),
            Tile::H5 => Ok(Tile::H6),
            Tile::H6 => Ok(Tile::H7),
            Tile::H7 => Ok(Tile::H8),
            Tile::H8 => Ok(Tile::H9),
            Tile::H9 => Err(AocError::ValueError("H9 is already highest".to_string())),
        }
    }
}

struct Map {
    width: usize,
    height: usize,
    data: Vec<Vec<Tile>>,
}

impl Map {
    fn parse(input: &str) -> Result<Map> {
        let parser = |_, _, c| Tile::try_from(c);
        let data = util::parse_tiles(&input, parser)?;
        let height = data.len();
        let width = if height > 0 { data[0].len() } else { 0 };
        Ok(Map {
            width,
            height,
            data,
        })
    }
}

fn get_trailscore_recursive(
    x: usize,
    y: usize,
    map: &Map,
    expected: Tile,
) -> (usize, HashSet<(usize, usize)>) {
    let mut set = HashSet::new();
    if map.data[y][x] != expected {
        return (0, set);
    }
    if expected == Tile::H9 {
        set.insert((x, y));
        return (1, set);
    }

    let next = expected.next().unwrap();
    let mut all_paths = 0;
    if x > 0 {
        let (paths, peaks) = get_trailscore_recursive(x - 1, y, map, next);
        set.extend(peaks);
        all_paths += paths;
    }
    if y > 0 {
        let (paths, peaks) = get_trailscore_recursive(x, y - 1, map, next);
        set.extend(peaks);
        all_paths += paths;
    }
    if x < map.width - 1 {
        let (paths, peaks) = get_trailscore_recursive(x + 1, y, map, next);
        set.extend(peaks);
        all_paths += paths;
    }
    if y < map.height - 1 {
        let (paths, peaks) = get_trailscore_recursive(x, y + 1, map, next);
        set.extend(peaks);
        all_paths += paths;
    }
    (all_paths, set)
}

fn get_map_score_and_rating(map: &Map) -> (usize, usize) {
    if map.height == 0 || map.width == 0 {
        return (0, 0);
    }

    let mut score = 0;
    let mut rating = 0;
    for y in 0..map.height {
        for x in 0..map.width {
            let (paths, set) = get_trailscore_recursive(x, y, map, Tile::H0);
            rating += paths;
            score += set.len();
        }
    }
    (score, rating)
}

pub fn run() -> Result<()> {
    println!("day 10");
    let path = PathBuf::from("./resources/day10.txt");
    let data = util::get_data_string(&path)?;
    let map = Map::parse(&data)?;
    let (score, rating) = get_map_score_and_rating(&map);
    println!("trail score: {score}");
    println!("trail rating: {rating}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_path() {
        let data = "55556
50547
51238
55559
55555";
        let map = Map::parse(&data).unwrap();
        let (score, rating) = get_map_score_and_rating(&map);
        assert_eq!(score, 1);
        assert_eq!(rating, 1);
    }

    #[test]
    fn test_two_paths_for_peak() {
        let data = "55556
50547
51238
54389
55675";
        let map = Map::parse(&data).unwrap();
        let (score, rating) = get_map_score_and_rating(&map);
        assert_eq!(score, 1);
        assert_eq!(rating, 2);
    }
}

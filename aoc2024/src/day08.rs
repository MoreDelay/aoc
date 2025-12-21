use anyhow::Result;
use std::{collections::HashMap, path::PathBuf};

use crate::util;

enum Tile {
    Empty(bool),
    Antenna(char, bool),
}

impl Tile {
    fn activate(&mut self) {
        let active = match self {
            Tile::Empty(active) => active,
            Tile::Antenna(_, active) => active,
        };
        *active = true;
    }
}

#[derive(Copy, Clone)]
struct Antenna {
    pos: (usize, usize),
}

struct Frequency {
    antennas: Vec<Antenna>,
}

struct Map {
    tiles: Vec<Vec<Tile>>,
    frequencies: HashMap<char, Frequency>,
}

impl Map {
    fn parse(input: &str) -> Result<Map> {
        let parser = |_, _, c| match c {
            '.' => Ok(Tile::Empty(false)),
            c => Ok(Tile::Antenna(c, false)),
        };
        let tiles = util::parse_tiles(input, parser)?;

        let mut frequencies = HashMap::new();
        for (y, row) in tiles.iter().enumerate() {
            for (x, t) in row.iter().enumerate() {
                let frequency = match t {
                    Tile::Antenna(c, _) => frequencies.entry(*c).or_insert_with(|| Frequency {
                        antennas: Vec::new(),
                    }),
                    _ => continue,
                };
                frequency.antennas.push(Antenna { pos: (x, y) });
            }
        }
        Ok(Map { tiles, frequencies })
    }
}

fn antinode_at(tiles: &[Vec<Tile>], near: Antenna, far: Antenna) -> Option<(usize, usize)> {
    let Antenna {
        pos: (x_near, y_near),
    } = near;
    let Antenna {
        pos: (x_far, y_far),
    } = far;
    let x_near = x_near as isize;
    let y_near = y_near as isize;
    let x_far = x_far as isize;
    let y_far = y_far as isize;

    let anti_x = x_near + x_near - x_far;
    let anti_y = y_near + y_near - y_far;

    let width = tiles[0].len() as isize;
    let height = tiles.len() as isize;

    if 0 <= anti_x && anti_x < width && 0 <= anti_y && anti_y < height {
        Some((anti_x as usize, anti_y as usize))
    } else {
        None
    }
}

fn set_antinodes(map: &mut Map) {
    for Frequency { antennas } in map.frequencies.values() {
        let n_antennas = antennas.len();
        for p in 0..n_antennas {
            for q in 0..n_antennas {
                if p == q {
                    continue;
                }
                if let Some((x, y)) = antinode_at(&map.tiles, antennas[p], antennas[q]) {
                    map.tiles[y][x].activate();
                }
                if let Some((x, y)) = antinode_at(&map.tiles, antennas[q], antennas[p]) {
                    map.tiles[y][x].activate();
                }
            }
        }
    }
}

fn set_harmonics_line(tiles: &mut [Vec<Tile>], mut near: Antenna, mut far: Antenna) {
    while let Some((x, y)) = antinode_at(tiles, near, far) {
        tiles[y][x].activate();
        far = near;
        near = Antenna { pos: (x, y) };
    }
}

fn set_resonant_harmonics(map: &mut Map) {
    for Frequency { antennas } in map.frequencies.values() {
        for &Antenna { pos: (x, y) } in antennas {
            map.tiles[y][x].activate();
        }
    }

    for Frequency { antennas } in map.frequencies.values() {
        let n_antennas = antennas.len();
        for p in 0..n_antennas {
            for q in 0..n_antennas {
                if p == q {
                    continue;
                }
                set_harmonics_line(&mut map.tiles, antennas[p], antennas[q]);
                set_harmonics_line(&mut map.tiles, antennas[q], antennas[p]);
            }
        }
    }
}

fn count_antinodes(map: &Map) -> usize {
    map.tiles
        .iter()
        .flatten()
        .map(|t| match t {
            Tile::Empty(active) => active,
            Tile::Antenna(_, active) => active,
        })
        .filter(|&&b| b)
        .count()
}

pub fn run() -> Result<()> {
    println!("day 08");
    let path = PathBuf::from("./resources/day08.txt");
    let data = util::get_data_string(&path)?;
    let mut map = Map::parse(&data)?;
    set_antinodes(&mut map);
    let antinodes = count_antinodes(&map);
    println!("antinodes: {antinodes}");
    set_resonant_harmonics(&mut map);
    let antinodes = count_antinodes(&map);
    println!("antinodes with resonant harmonics: {antinodes}");
    Ok(())
}

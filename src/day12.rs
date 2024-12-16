use anyhow::Result;
use std::{collections::VecDeque, path::PathBuf};

use crate::util;

#[derive(Copy, Clone, Debug)]
struct Tile {
    name: char,
    visited: bool,
}

struct Plot {
    width: usize,
    height: usize,
    data: Vec<Vec<Tile>>,
}

fn parse_plot(input: &str) -> Result<Plot> {
    let parser = |_, _, c| {
        Ok(Tile {
            name: c,
            visited: false,
        })
    };
    let data = util::parse_tiles(input, parser)?;
    let height = data.len();
    let width = if height > 0 { data[0].len() } else { 0 };
    Ok(Plot {
        width,
        height,
        data,
    })
}

fn get_price_at(pos: (usize, usize), plot: &mut Plot) -> usize {
    let (x, y) = pos;
    assert!(x < plot.width && y < plot.height);
    let expected = match plot.data[y][x] {
        Tile { visited, .. } if visited => return 0,
        Tile { name, .. } => name,
    };

    let mut perimeter = 0;
    let mut fields = 0;

    let mut queue = VecDeque::new();
    queue.push_back(Some(pos));
    while !queue.is_empty() {
        let next = queue.pop_front().expect("checked not empty");
        let tile = next.map(|(x, y)| &mut plot.data[y][x]);
        match tile {
            None => {
                perimeter += 1;
                continue;
            }
            Some(Tile { name, .. }) if *name != expected => {
                perimeter += 1;
                continue;
            }
            Some(Tile { visited, .. }) if !*visited => {
                *visited = true;
                fields += 1;
            }
            Some(_) => {
                continue;
            }
        }

        let (x, y) = next.expect("checked in match");
        queue.push_back((x > 0).then(|| (x - 1, y)));
        queue.push_back((y > 0).then(|| (x, y - 1)));
        queue.push_back((x < plot.width - 1).then(|| (x + 1, y)));
        queue.push_back((y < plot.height - 1).then(|| (x, y + 1)));
    }

    perimeter * fields
}

fn get_perimeter_price(mut plot: Plot) -> usize {
    let mut price = 0;
    for y in 0..plot.height {
        for x in 0..plot.width {
            price += get_price_at((x, y), &mut plot);
        }
    }
    price
}

pub fn run() -> Result<()> {
    println!("day 12");
    let path = PathBuf::from("./resources/day12.txt");
    let data = util::get_data_string(&path)?;
    let plot = parse_plot(&data)?;
    let price = get_perimeter_price(plot);
    println!("price for perimeter: {price}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_plot() {
        let input = "AAAA
BBCD
BBCC
EEEC";
        let plot = parse_plot(input).unwrap();
        let price = get_perimeter_price(plot);
        assert_eq!(price, 140);
    }

    #[test]
    fn test_plot_with_holes() {
        let input = "OOOOO
OXOXO
OOOOO
OXOXO
OOOOO";
        let plot = parse_plot(input).unwrap();
        let price = get_perimeter_price(plot);
        assert_eq!(price, 772);
    }
}

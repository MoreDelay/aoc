use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::util::{Grid, GridParseError, Pos};

#[derive(Debug)]
enum Tile {
    Empty,
    Splitter,
    Start,
}

impl std::fmt::Display for Tile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tile::Empty => write!(f, "."),
            Tile::Splitter => write!(f, "^"),
            Tile::Start => write!(f, "S"),
        }
    }
}

#[derive(Debug, Error)]
pub enum TileParseError {
    #[error("unexpected character, expect one of '.^S', got '{0}'")]
    UnexpectedCharacter(char),
    #[error("found two start positions, at {0} and {1}")]
    AmbiguousStart(Pos, Pos),
}

impl Tile {
    fn parse(c: char) -> Result<Self, TileParseError> {
        match c {
            '.' => Ok(Tile::Empty),
            '^' => Ok(Tile::Splitter),
            'S' => Ok(Tile::Start),
            _ => Err(TileParseError::UnexpectedCharacter(c)),
        }
    }
}

#[derive(Debug)]
struct TachyonManifold {
    grid: Grid<Tile>,
    start: Pos,
}

#[derive(Debug, Error)]
pub enum TachyonManifoldParseError {
    #[error("could not parse grid from input")]
    GridParseError(#[from] GridParseError<TileParseError>),
    #[error("no start in input grid")]
    MissingStart,
}

impl TachyonManifold {
    fn parse(input: impl std::io::BufRead) -> Result<Self, TachyonManifoldParseError> {
        let mut start = None;
        let parse = |x, y, c| {
            let tile = Tile::parse(c)?;
            if matches!(tile, Tile::Start) {
                let pos = Pos(x, y);
                if let Some(start) = start.take() {
                    return Err(TileParseError::AmbiguousStart(start, pos));
                }
                start = Some(pos);
            }
            Ok(tile)
        };
        let grid = Grid::parse(input, parse)?;

        let Some(start) = start else {
            return Err(TachyonManifoldParseError::MissingStart);
        };
        Ok(Self { grid, start })
    }

    fn count_splits(&self) -> usize {
        let mut beams = BTreeSet::from_iter([self.start.0]);
        let mut count = 0;

        for y in 1..self.grid.height() {
            beams = beams
                .into_iter()
                .flat_map(
                    |x| match self.grid.get(x, y).expect("beam stays within grid") {
                        Tile::Empty => either::Left(std::iter::once(x)),
                        Tile::Splitter => {
                            count += 1;
                            either::Right(std::iter::once(x - 1).chain(std::iter::once(x + 1)))
                        }
                        Tile::Start => unreachable!(),
                    },
                )
                .collect();
        }
        count
    }

    fn follow_beam_to_splitter(&self, start: Pos) -> Option<Pos> {
        let mut pos = start;
        loop {
            let Pos(x, y) = pos;
            match self.grid.get(x, y)? {
                Tile::Splitter => return Some(pos),
                _ => pos = Pos(x, y + 1),
            }
        }
    }

    fn count_timelines(&self) -> usize {
        use std::collections::btree_map::Entry;
        let mut splits = BTreeMap::new();

        for y in (0..self.grid.height()).rev() {
            for x in 0..self.grid.width() {
                let tile = self.grid.get(x, y).expect("we are within the grid");
                if !matches!(tile, Tile::Splitter) {
                    continue;
                }

                let left = self
                    .follow_beam_to_splitter(Pos(x - 1, y))
                    .map(|pos| splits.get(&pos).expect("already visited"))
                    .copied()
                    .unwrap_or(1);
                let right = self
                    .follow_beam_to_splitter(Pos(x + 1, y))
                    .map(|pos| splits.get(&pos).expect("already visited"))
                    .copied()
                    .unwrap_or(1);

                let pos = Pos(x, y);
                match splits.entry(pos) {
                    Entry::Vacant(entry) => entry.insert(left + right),
                    Entry::Occupied(_) => unreachable!(),
                };
            }
        }

        self.follow_beam_to_splitter(self.start)
            .map(|pos| splits.get(&pos).expect("already visited"))
            .copied()
            .unwrap_or(1)
    }
}

#[derive(Debug, Error)]
pub enum Day07Error {
    #[error("could not open file")]
    FileError(#[from] std::io::Error),
    #[error("could not parse input")]
    ParseError(#[from] TachyonManifoldParseError),
}

pub fn day07() -> Result<(), Day07Error> {
    let path = std::path::PathBuf::from("resources/day07.txt");
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let manifold = TachyonManifold::parse(reader)?;
    let splits = manifold.count_splits();
    println!("day07: number of splits is {splits}");
    let timelines = manifold.count_timelines();
    println!("day07: number of timelines is {timelines}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    static EXAMPLE: &str = ".......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
...............";

    #[test]
    fn test_input_01() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let manifold = TachyonManifold::parse(cursor)?;
        let count = manifold.count_splits();
        assert_eq!(count, 21);
        Ok(())
    }

    #[test]
    fn test_input_02() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let manifold = TachyonManifold::parse(cursor)?;
        let count = manifold.count_timelines();
        assert_eq!(count, 40);
        Ok(())
    }
}

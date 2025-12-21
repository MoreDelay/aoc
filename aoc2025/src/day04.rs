use itertools::iproduct;
use thiserror::Error;

use crate::util::{self, GridParseError};

#[derive(Debug, Clone, Copy)]
enum GridElement {
    Empty,
    PaperRoll,
}

#[derive(Debug, Error)]
pub enum GridElementParseError {
    #[error("unsupported character, expected '.' or '@', got '{0}'")]
    UnknownElement(char),
}

impl GridElement {
    fn parse(c: char) -> Result<GridElement, GridElementParseError> {
        match c {
            '.' => Ok(GridElement::Empty),
            '@' => Ok(GridElement::PaperRoll),
            _ => Err(GridElementParseError::UnknownElement(c)),
        }
    }
}

struct WarehouseGrid(util::Grid<GridElement>);

impl WarehouseGrid {
    fn parse(
        input: impl std::io::BufRead,
    ) -> Result<Self, util::GridParseError<GridElementParseError>> {
        let parse = |_, _, c| GridElement::parse(c);
        let grid = util::Grid::parse(input, parse)?;
        Ok(Self(grid))
    }

    fn at(&self, x: isize, y: isize) -> GridElement {
        if x < 0 || y < 0 {
            return GridElement::Empty;
        }
        let x = x as usize;
        let y = y as usize;
        self.0.get(x, y).copied().unwrap_or(GridElement::Empty)
    }

    fn iter_freestanding_roll_positions(&self) -> impl Iterator<Item = (usize, usize)> {
        iproduct!(0..self.0.width(), 0..self.0.height()).filter(|&(x, y)| {
            let x = x as isize;
            let y = y as isize;
            if matches!(self.at(x, y), GridElement::Empty) {
                return false;
            }

            let rolls_around = iproduct!(-1..=1, -1..=1)
                .filter(|(i, j)| matches!(self.at(x + i, y + j), GridElement::PaperRoll))
                .count();

            // we compare against 5 as we counted the roll in question too
            rolls_around < 5
        })
    }

    fn count_removable_rolls(mut self) -> usize {
        let mut count = 0;
        let mut last_count = None;

        while last_count.is_none_or(|c| c != count) {
            last_count = Some(count);
            let positions = self.iter_freestanding_roll_positions().collect::<Vec<_>>();

            for (x, y) in positions {
                let element = self
                    .0
                    .get_mut(x, y)
                    .expect("only valid positions are returned");
                assert!(matches!(*element, GridElement::PaperRoll));

                *element = GridElement::Empty;
                count += 1;
            }
        }

        count
    }
}

#[derive(Debug, Error)]
pub enum Day04Error {
    #[error("could not open file")]
    FileError(#[from] std::io::Error),
    #[error("could not parse input")]
    ParseError(#[from] GridParseError<GridElementParseError>),
}

pub fn day04() -> Result<(), Day04Error> {
    let path = std::path::PathBuf::from("resources/day04.txt");
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let grid = WarehouseGrid::parse(reader)?;
    let positions = grid.iter_freestanding_roll_positions().count();
    println!("day04: number of freestanding rolls is {positions}");
    let removable = grid.count_removable_rolls();
    println!("day04: number of removable rolls is {removable}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    static EXAMPLE: &str = "..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@.";

    #[test]
    fn test_input_01() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let grid = WarehouseGrid::parse(cursor)?;
        let count = grid.iter_freestanding_roll_positions().count();
        assert_eq!(count, 13);
        Ok(())
    }

    #[test]
    fn test_input_02() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let grid = WarehouseGrid::parse(cursor)?;
        let count = grid.count_removable_rolls();
        assert_eq!(count, 43);
        Ok(())
    }
}

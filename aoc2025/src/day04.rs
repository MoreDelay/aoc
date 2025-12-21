use itertools::iproduct;
use thiserror::Error;

#[derive(Debug, Clone, Copy)]
enum GridElement {
    Empty,
    PaperRoll,
}

#[derive(Debug, Error)]
enum GridElementParseError {
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

struct WarehouseGrid {
    elements: Vec<GridElement>,
    width: usize,
    height: usize,
}

#[derive(Debug, Error)]
enum WarehouseGridParseError {
    #[error("error while reading")]
    ReadError(#[from] std::io::Error),
    #[error("wrong format on line {0}")]
    WrongFormat(usize, #[source] GridElementParseError),
    #[error("input has not any element")]
    NoDataForGrid,
    #[error("input is not a regular grid at line {0}")]
    NotAGrid(usize),
}

impl WarehouseGrid {
    fn parse(input: impl std::io::BufRead) -> Result<Self, WarehouseGridParseError> {
        let mut elements = Vec::new();
        let mut width: Option<usize> = None;
        let mut height = 0;

        for (i, line) in input.lines().enumerate() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            match width {
                Some(width) => {
                    if width != line.len() {
                        return Err(WarehouseGridParseError::NotAGrid(i));
                    }
                }
                None => width = Some(line.len()),
            }

            for c in line.chars() {
                let element = GridElement::parse(c)
                    .map_err(|e| WarehouseGridParseError::WrongFormat(i, e))?;
                elements.push(element);
            }

            height += 1;
        }

        if height == 0 || width.is_none() {
            return Err(WarehouseGridParseError::NoDataForGrid);
        }
        let width = width.expect("checked above");
        assert_eq!(elements.len(), width * height);

        Ok(Self {
            elements,
            width,
            height,
        })
    }

    fn at(&self, x: isize, y: isize) -> GridElement {
        let x_in_range = 0 <= x && x < self.width as isize;
        let y_in_range = 0 <= y && y < self.height as isize;
        let in_range = x_in_range && y_in_range;
        if !in_range {
            return GridElement::Empty;
        }

        let x = x as usize;
        let y = y as usize;

        let index = y * self.width + x;
        self.elements
            .get(index)
            .copied()
            .unwrap_or(GridElement::Empty)
    }

    fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut GridElement> {
        let in_range = x < self.width && y < self.height;
        if !in_range {
            return None;
        }

        let index = y * self.width + x;
        self.elements.get_mut(index)
    }

    fn iter_freestanding_roll_positions(&self) -> impl Iterator<Item = (usize, usize)> {
        iproduct!(0..self.width, 0..self.height).filter(|&(x, y)| {
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

pub fn day04() -> anyhow::Result<()> {
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

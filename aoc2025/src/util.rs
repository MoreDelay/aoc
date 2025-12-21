use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pos(pub usize, pub usize);

impl std::fmt::Display for Pos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

#[derive(Debug, Clone)]
pub struct Grid<T> {
    tiles: Vec<T>,
    width: usize,
    height: usize,
}

#[derive(Debug, Error)]
pub enum GridParseError<E: std::error::Error> {
    #[error("error while reading")]
    ReadError(#[from] std::io::Error),
    #[error("wrong format on row {0}, col {1}")]
    WrongFormat(usize, usize, #[source] E),
    #[error("input has not any element")]
    NoDataForGrid,
    #[error("input is not a regular grid at line {0}")]
    NotAGrid(usize),
}

impl<T> Grid<T> {
    pub fn parse<E: std::error::Error>(
        input: impl std::io::BufRead,
        mut parse_tile: impl FnMut(usize, usize, char) -> Result<T, E>,
    ) -> Result<Self, GridParseError<E>> {
        let mut tiles = Vec::new();
        let mut width: Option<usize> = None;
        let mut height = 0;

        for (y, line) in input.lines().enumerate() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            match width {
                Some(width) => {
                    if width != line.len() {
                        return Err(GridParseError::NotAGrid(y));
                    }
                }
                None => width = Some(line.len()),
            }

            for (x, c) in line.chars().enumerate() {
                let element =
                    parse_tile(x, y, c).map_err(|e| GridParseError::WrongFormat(x, y, e))?;
                tiles.push(element);
            }

            height += 1;
        }

        if height == 0 || width.is_none() {
            return Err(GridParseError::NoDataForGrid);
        }
        let width = width.expect("checked above");
        assert_eq!(tiles.len(), width * height);

        Ok(Self {
            tiles,
            width,
            height,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    fn index(&self, x: usize, y: usize) -> Option<usize> {
        let in_range = x < self.width && y < self.height;
        match in_range {
            true => Some(y * self.width + x),
            false => None,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> Option<&T> {
        let index = self.index(x, y)?;
        self.tiles.get(index)
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut T> {
        let index = self.index(x, y)?;
        self.tiles.get_mut(index)
    }
}

impl<T> std::fmt::Display for Grid<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self.tiles.iter();
        for t in iter.by_ref().take(self.width) {
            t.fmt(f)?;
        }
        for _ in 1..self.height {
            writeln!(f)?;
            for t in iter.by_ref().take(self.width) {
                t.fmt(f)?;
            }
        }
        Ok(())
    }
}

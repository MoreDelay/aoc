use std::num::ParseIntError;

use itertools::Itertools;
use thiserror::Error;

#[derive(Debug, Copy, Clone)]
pub struct Tile {
    x: usize,
    y: usize,
}

#[derive(Debug, Error)]
pub enum TileParseError {
    #[error("expect format '<number>,<number>', got '{0}'")]
    WrongFormat(String),
    #[error("could not parse as a number")]
    NotANumber(#[from] ParseIntError),
}

impl Tile {
    fn parse(input: &str) -> Result<Self, TileParseError> {
        let comma = input
            .find(',')
            .ok_or_else(|| TileParseError::WrongFormat(input.to_string()))?;
        let (x, y) = input.split_at(comma);
        let y = &y[1..]; // remove comma
        let x = x.parse()?;
        let y = y.parse()?;
        Ok(Self { x, y })
    }
}

pub struct Tiles(Vec<Tile>);

#[derive(Debug, Error)]
pub enum TilesParseError {
    #[error("error while reading")]
    ReadError(#[from] std::io::Error),
    #[error("could not parse line {0} as a junction box")]
    TileParseError(usize, #[source] TileParseError),
    #[error("not enough tiles to form a rectangle, only got {0}")]
    NotEnoughTiles(usize),
}

impl Tiles {
    fn parse(input: impl std::io::BufRead) -> Result<Self, TilesParseError> {
        let tiles = input
            .lines()
            .enumerate()
            .map(|(i, line)| Tile::parse(&line?).map_err(|e| TilesParseError::TileParseError(i, e)))
            .collect::<Result<Vec<_>, _>>()?;
        if tiles.len() < 2 {
            return Err(TilesParseError::NotEnoughTiles(tiles.len()));
        }
        Ok(Self(tiles))
    }
}

#[derive(Debug)]
struct Rectangle(Tile, Tile);

impl Rectangle {
    fn area(&self) -> usize {
        let Self(a, b) = self;
        let dx = a.x.abs_diff(b.x) + 1;
        let dy = a.y.abs_diff(b.y) + 1;
        dx * dy
    }

    fn aabb(&self) -> AaBb {
        AaBb::new(self.0, self.1)
    }
}

impl Tiles {
    fn find_largest_rectangle(&self) -> usize {
        let Tiles(tiles) = self;
        let n = tiles.len();

        let mut area = 0;
        for i in 0..n {
            for j in i..n {
                let rect = Rectangle(tiles[i], tiles[j]);
                area = area.max(rect.area());
            }
        }
        area
    }
}

#[derive(Debug, Error)]
pub enum RopeError {
    #[error("provided input is not an axis-aligned rope, can not go from {0:?} to {1:?}")]
    NotARope(Tile, Tile),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    fn between_tiles(from: &Tile, to: &Tile) -> Option<Self> {
        use std::cmp::Ordering;
        let Tile { x: x1, y: y1 } = from;
        let Tile { x: x2, y: y2 } = to;
        match (x1.cmp(x2), y1.cmp(y2)) {
            (Ordering::Less, Ordering::Less) => None,
            (Ordering::Less, Ordering::Equal) => Some(Self::East),
            (Ordering::Less, Ordering::Greater) => None,
            (Ordering::Equal, Ordering::Less) => Some(Self::South),
            (Ordering::Equal, Ordering::Equal) => None,
            (Ordering::Equal, Ordering::Greater) => Some(Self::North),
            (Ordering::Greater, Ordering::Less) => None,
            (Ordering::Greater, Ordering::Equal) => Some(Self::West),
            (Ordering::Greater, Ordering::Greater) => None,
        }
    }
}

#[derive(Debug)]
struct Segment {
    start: Tile,
    end: Tile,
}

impl Segment {
    fn new(start: Tile, end: Tile) -> Option<Self> {
        Direction::between_tiles(&start, &end)?;
        Some(Self { start, end })
    }

    fn dir(&self) -> Direction {
        Direction::between_tiles(&self.start, &self.end).expect("valid by construction")
    }
}

#[derive(Debug)]
struct AaBb {
    hi: Tile,
    lo: Tile,
}

impl AaBb {
    fn new(a: Tile, b: Tile) -> Self {
        let lo = Tile {
            x: a.x.min(b.x),
            y: a.y.min(b.y),
        };
        let hi = Tile {
            x: a.x.max(b.x),
            y: a.y.max(b.y),
        };
        Self { lo, hi }
    }

    fn intersected_by(&self, seg: &Segment) -> bool {
        let AaBb { hi, lo } = self;

        match seg.dir() {
            Direction::North | Direction::South => {
                let fix_x = seg.start.x;
                assert_eq!(fix_x, seg.end.x);
                let min_y = seg.start.y.min(seg.end.y);
                let max_y = seg.start.y.max(seg.end.y);

                let inside_x = lo.x < fix_x && fix_x < hi.x;
                let cross_y = min_y <= lo.y && lo.y < max_y || min_y < hi.y && hi.y <= max_y;
                inside_x && cross_y
            }
            Direction::East | Direction::West => {
                let min_x = seg.start.x.min(seg.end.x);
                let max_x = seg.start.x.max(seg.end.x);
                let fix_y = seg.start.y;
                assert_eq!(fix_y, seg.end.y);

                let inside_y = lo.y < fix_y && fix_y < hi.y;
                let cross_x = min_x <= lo.x && lo.x < max_x || min_x < hi.x && hi.x <= max_x;
                inside_y && cross_x
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Rotation {
    Clockwise,
    CounterClockwise,
}

enum RotationError {
    SameDir,
    Inverse,
}

impl Rotation {
    fn from_dirs(initial: Direction, then: Direction) -> Result<Self, RotationError> {
        match (initial, then) {
            (Direction::North, Direction::North) => Err(RotationError::SameDir),
            (Direction::North, Direction::East) => Ok(Self::Clockwise),
            (Direction::North, Direction::South) => Err(RotationError::Inverse),
            (Direction::North, Direction::West) => Ok(Self::CounterClockwise),
            (Direction::East, Direction::North) => Ok(Self::CounterClockwise),
            (Direction::East, Direction::East) => Err(RotationError::SameDir),
            (Direction::East, Direction::South) => Ok(Self::Clockwise),
            (Direction::East, Direction::West) => Err(RotationError::Inverse),
            (Direction::South, Direction::North) => Err(RotationError::Inverse),
            (Direction::South, Direction::East) => Ok(Self::CounterClockwise),
            (Direction::South, Direction::South) => Err(RotationError::SameDir),
            (Direction::South, Direction::West) => Ok(Self::Clockwise),
            (Direction::West, Direction::North) => Ok(Self::Clockwise),
            (Direction::West, Direction::East) => Err(RotationError::Inverse),
            (Direction::West, Direction::South) => Ok(Self::CounterClockwise),
            (Direction::West, Direction::West) => Err(RotationError::SameDir),
        }
    }
}

#[derive(Debug)]
struct RotationTracker {
    count: isize,
    last_dir: Direction,
}

impl RotationTracker {
    fn new(initial_dir: Direction) -> Self {
        Self {
            count: 0,
            last_dir: initial_dir,
        }
    }

    fn add_dir(&mut self, dir: Direction) {
        let rotation = Rotation::from_dirs(self.last_dir, dir);
        match rotation {
            Ok(Rotation::Clockwise) => self.count += 1,
            Ok(Rotation::CounterClockwise) => self.count -= 1,
            Err(RotationError::SameDir) => (),
            Err(RotationError::Inverse) if self.count > 0 => self.count -= 2,
            Err(RotationError::Inverse) => self.count += 2,
        }
    }
}

struct Rope {
    tiles: Vec<Tile>,
}

impl Rope {
    fn new(tiles: Tiles) -> Result<Self, RopeError> {
        let Tiles(tiles) = tiles;
        let [start, second] = &tiles[..2] else {
            unreachable!()
        };

        let dir =
            Direction::between_tiles(start, second).ok_or(RopeError::NotARope(*start, *second))?;

        let mut tracker = RotationTracker::new(dir);
        let mut from = start;
        for to in tiles[1..].iter().chain(std::iter::once(start)) {
            let dir = Direction::between_tiles(from, to).ok_or(RopeError::NotARope(*from, *to))?;
            tracker.add_dir(dir);
            from = to;
        }

        Ok(Self { tiles })
    }

    fn find_largest_rectangle_inside(&self) -> usize {
        let mut cur_area = 1;

        let n = self.tiles.len();
        let rectangles = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .map(|(i, j)| Rectangle(self.tiles[i], self.tiles[j]));
        for rect in rectangles {
            let aabb = rect.aabb();
            let invalid = self
                .tiles
                .iter()
                .circular_tuple_windows()
                .any(|(from, to)| {
                    let seg = Segment::new(*from, *to).expect("valid by construction");
                    aabb.intersected_by(&seg)
                });
            if !invalid {
                cur_area = cur_area.max(rect.area());
            }
        }

        cur_area
    }
}

#[derive(Debug, Error)]
pub enum Day09Error {
    #[error("could not open file")]
    FileError(#[from] std::io::Error),
    #[error("could not parse input")]
    ParseError(#[from] TilesParseError),
    #[error("could not construct rope of tiles")]
    RopeFailed(#[from] RopeError),
}

pub fn day09() -> Result<(), Day09Error> {
    let path = std::path::PathBuf::from("resources/day09.txt");
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let tiles = Tiles::parse(reader)?;
    let area = tiles.find_largest_rectangle();
    println!("day09: the largest rectangle has area {area}");
    let rope = Rope::new(tiles)?;
    let area = rope.find_largest_rectangle_inside();
    println!("day09: the largest rectangle inside the rope has area {area}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    static EXAMPLE: &str = "7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3";

    #[test]
    fn test_input_01() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let tiles = Tiles::parse(cursor)?;
        let area = tiles.find_largest_rectangle();
        assert_eq!(area, 50);
        Ok(())
    }

    #[test]
    fn test_input_02() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let tiles = Tiles::parse(cursor)?;
        let rope = Rope::new(tiles)?;
        let area = rope.find_largest_rectangle_inside();
        assert_eq!(area, 24);
        Ok(())
    }
}

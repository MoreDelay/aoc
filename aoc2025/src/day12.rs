use std::{
    collections::{HashMap, VecDeque},
    io::{Cursor, Read},
};

use bitvec::prelude as bv;
use thiserror::Error;

use crate::util::{self, Grid, Pos};

#[derive(Debug, Clone, Copy)]
enum Tile {
    Empty,
    Filled,
}

#[derive(Debug, Error)]
pub enum TileParseError {
    #[error("unexpected character '{2}' found at location ({0}, {1})")]
    UnexpectedTile(usize, usize, char),
}

impl Tile {
    fn parse(x: usize, y: usize, c: char) -> Result<Self, TileParseError> {
        match c {
            '.' => Ok(Tile::Empty),
            '#' => Ok(Tile::Filled),
            _ => Err(TileParseError::UnexpectedTile(x, y, c)),
        }
    }
}

#[derive(Debug)]
struct Shape {
    #[expect(unused)]
    number: usize,
    grid: Grid<Tile>,
}

#[derive(Debug, Error)]
pub enum ShapeParseError {
    #[error("header missing from shape description")]
    MissingHeader,
    #[error("expect header format '<number>:', got {0}")]
    WrongHeader(String),
    #[error("could not parse header number")]
    HeaderNumber(#[from] std::num::ParseIntError),
    #[error("could not parse shape")]
    GridError(#[from] util::GridParseError<TileParseError>),
}

impl Shape {
    fn parse(input: &str) -> Result<Self, ShapeParseError> {
        let Some((header, shape)) = input.split_once('\n') else {
            return Err(ShapeParseError::MissingHeader);
        };

        let Some((number, empty)) = header.split_once(':') else {
            return Err(ShapeParseError::WrongHeader(header.to_string()));
        };
        if !empty.is_empty() {
            return Err(ShapeParseError::WrongHeader(header.to_string()));
        }
        let number = number.parse()?;

        let shape = Cursor::new(shape);
        let grid = Grid::parse(shape, Tile::parse)?;
        Ok(Self { number, grid })
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ShapeCounts(Vec<usize>);

struct Region {
    width: usize,
    height: usize,
    shape_counts: ShapeCounts,
}

#[derive(Debug, Error)]
pub enum RegionParseError {
    #[error("expected prefix format '<number>x<number>:( <number>)+', got {0}")]
    WrongFormat(String),
    #[error("expected prefix format '<number>x<number>:', got {0}")]
    Prefix(String),
    #[error("could not parse a number")]
    Number(#[from] std::num::ParseIntError),
    #[error("we got {1} shapes, but only count of {0} shapes are given")]
    MissingCount(usize, usize),
}

impl Region {
    fn parse(input: &str, n_shapes: usize) -> Result<Self, RegionParseError> {
        let Some((prefix, counts)) = input.split_once(' ') else {
            return Err(RegionParseError::WrongFormat(input.to_string()));
        };

        let Some((size, empty)) = prefix.split_once(':') else {
            return Err(RegionParseError::Prefix(prefix.to_string()));
        };
        if !empty.is_empty() {
            return Err(RegionParseError::Prefix(prefix.to_string()));
        }
        let Some((width, height)) = size.split_once('x') else {
            return Err(RegionParseError::Prefix(prefix.to_string()));
        };
        let width = width.parse()?;
        let height = height.parse()?;

        let shape_counts = counts
            .split(' ')
            .map(|c| c.parse())
            .collect::<Result<Vec<_>, _>>()?;
        let n_counts = shape_counts.len();
        if n_counts != n_shapes {
            return Err(RegionParseError::MissingCount(n_counts, n_shapes));
        }
        let shape_counts = ShapeCounts(shape_counts);

        Ok(Self {
            width,
            height,
            shape_counts,
        })
    }

    fn shapes_fit_naive(&self, shapes: &[Shape]) -> bool {
        assert_eq!(self.shape_counts.0.len(), shapes.len());

        let blocked = self
            .shape_counts
            .0
            .iter()
            .zip(shapes.iter())
            .map(|(c, s)| {
                c * s
                    .grid
                    .iter()
                    .filter(|(_, _, t)| matches!(t, Tile::Filled))
                    .count()
            })
            .sum();
        let open = self.width * self.height;
        open >= blocked
    }

    fn shapes_fit(&self, shapes: &[Shape], cache: &mut Cache) -> bool {
        assert_eq!(self.shape_counts.0.len(), shapes.len());

        #[derive(Debug)]
        struct DfsPartition {
            connected: Option<ConnectedPartition>,
            iter: PlacementConnectedPartitionIterator,
        }

        impl DfsPartition {
            fn new(placement: Placement) -> Self {
                let mut iter = placement.iter_connected_partitions();
                let connected = iter.next();
                DfsPartition { connected, iter }
            }
        }

        #[derive(Debug)]
        struct DfsPlacement {
            iter: OpenShapePlacementIterator,
            partition: Option<DfsPartition>,
        }

        impl DfsPlacement {
            fn new(parent: &ConnectedPartition, shape: &Shape) -> Self {
                let mut iter = parent.iter_shape_placements(shape);
                let partition = iter.next().clone().map(DfsPartition::new);
                DfsPlacement { iter, partition }
            }
        }

        #[derive(Debug)]
        struct DfsState {
            parent: ConnectedPartition,
            shape_index: usize,
            counts: ShapeCounts,
            placement: Option<DfsPlacement>,
        }

        let mut stack = {
            let full_region = ConnectedPartition::new(self.width, self.height);
            let counts = self.shape_counts.clone();
            let Some(shape) = shapes.get(0) else {
                // edge case: no shapes provided, so it always fits
                return true;
            };
            let placement = Some(DfsPlacement::new(&full_region, shape));
            let init = DfsState {
                parent: full_region,
                shape_index: 0,
                counts,
                placement,
            };
            Vec::from_iter([init])
        };

        let mut iterations = 0;
        const MAX_ITER: usize = 100_000_000;

        'stack: while let Some(mut state) = stack.pop() {
            let mut child_counts = state.counts.clone();
            let shape_counter = &mut child_counts.0[state.shape_index];
            match shape_counter.checked_sub(1) {
                Some(reduced) => *shape_counter = reduced,
                // can not reduce this shape any further, skip directly to next shape
                None => state.placement = None,
            };

            'placement: while let Some(dfs_placement) = state.placement.as_mut() {
                'partition: while let Some(dfs_partition) = dfs_placement.partition.as_mut() {
                    iterations += 1;
                    assert!(iterations < MAX_ITER);
                    // check if the current placement fits
                    let Some(connected) = dfs_partition.connected.take() else {
                        // exhausted iterator
                        dfs_placement.partition = None;
                        break 'partition;
                    };

                    let remaining_shapes = child_counts.0.iter().sum::<usize>();
                    if remaining_shapes == 0 {
                        // stop condition
                        let key = (state.parent, state.counts);
                        cache.inner.insert(key, true);
                        continue 'stack;
                    }

                    let child_key = (connected, child_counts);
                    let cache_entry = cache.inner.get(&child_key).cloned();
                    let (connected, key_counts) = child_key;
                    child_counts = key_counts; // move counts back for test in next iteration

                    match cache_entry {
                        Some(true) => {
                            // confirmed that we fit
                            let key = (state.parent, state.counts);
                            cache.inner.insert(key, true);
                            continue 'stack;
                        }
                        Some(false) => {
                            // try next connected partition
                            dfs_partition.connected = dfs_partition.iter.next();
                            continue 'partition;
                        }
                        None => {
                            // not yet tested
                            dfs_partition.connected = Some(connected.clone());
                            stack.push(state);

                            let next = DfsState {
                                parent: connected,
                                shape_index: 0,
                                counts: child_counts,
                                placement: None,
                            };
                            stack.push(next);
                            continue 'stack;
                        }
                    }
                }

                // create new placement and connected partition iterator
                assert!(dfs_placement.partition.is_none());

                let Some(placement) = dfs_placement.iter.next() else {
                    // iterator exhausted, try to place next shape
                    state.placement = None;
                    break 'placement;
                };
                dfs_placement.partition = Some(DfsPartition::new(placement));
            }

            // try next shape and create new placement iterator
            assert!(state.placement.is_none());

            state.shape_index += 1;
            if state.shape_index >= shapes.len() {
                // exhausted shapes, confirmed not fitting
                let key = (state.parent, state.counts);
                cache.inner.insert(key, false);
                continue 'stack;
            }

            let shape = &shapes[state.shape_index];
            state.placement = Some(DfsPlacement::new(&state.parent, shape));
            stack.push(state);
        }

        // cache is now full with all results
        let full_region = ConnectedPartition::new(self.width, self.height);
        let counts = self.shape_counts.clone();
        let key = (full_region, counts);
        cache.inner.get(&key).copied().expect("cache completed")
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct BitShape {
    width: usize,
    height: usize,
    occupied: bv::BitVec,
}

impl std::fmt::Display for BitShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, bit) in self.occupied.iter().enumerate() {
            if i > 0 && i % self.width == 0 {
                write!(f, "\n")?;
            }
            match *bit {
                true => write!(f, "#")?,
                false => write!(f, ".")?,
            }
        }
        Ok(())
    }
}

impl std::ops::Index<(usize, usize)> for BitShape {
    type Output = bool;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (x, y) = index;
        assert!(x < self.width && y < self.height);
        &self.occupied[y * self.width + x]
    }
}

impl BitShape {
    fn get(&self, x: usize, y: usize) -> Option<bool> {
        let inside = x < self.width && y < self.height;
        match inside {
            true => Some(self.occupied[y * self.width + x]),
            false => None,
        }
    }

    fn set(&mut self, x: usize, y: usize, v: bool) -> Option<bool> {
        let inside = x < self.width && y < self.height;
        match inside {
            true => {
                let index = y * self.width + x;
                let prev = self.occupied[index];
                self.occupied.set(index, v);
                Some(prev)
            }
            false => None,
        }
    }

    fn rotate(self) -> Self {
        let prev = &self;
        let occupied = (0..self.width)
            .rev()
            .flat_map(|x| (0..self.height).map(move |y| prev[(x, y)]));
        let occupied = bv::BitVec::from_iter(occupied);

        BitShape {
            width: self.height,
            height: self.width,
            occupied,
        }
    }

    fn flip(self) -> Self {
        let prev = &self;
        let occupied = (0..self.height)
            .rev()
            .flat_map(|y| (0..self.width).map(move |x| prev[(x, y)]));
        let occupied = bv::BitVec::from_iter(occupied);

        BitShape {
            width: self.width,
            height: self.height,
            occupied,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct OpenShape(BitShape);

impl std::fmt::Display for OpenShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::ops::Deref for OpenShape {
    type Target = BitShape;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for OpenShape {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl OpenShape {}

#[derive(Debug, Clone)]
struct PlacementConnectedPartitionIterator {
    placement: Placement,
    visited: BitShape,
    last_pos: Option<Pos>,
}

impl PlacementConnectedPartitionIterator {
    fn new(shape: Placement) -> Self {
        let width = shape.0.width;
        let height = shape.0.height;
        let visited = bv::bitvec![0; width * height];
        let visited = BitShape {
            width,
            height,
            occupied: visited,
        };
        Self {
            placement: shape,
            visited,
            last_pos: None,
        }
    }

    fn bfs_connected_partition(&mut self, start: Pos) -> ConnectedPartition {
        #[derive(Debug)]
        struct BfsState {
            pos: Pos,
        }
        const MAX_ITER: usize = 1_000_000;

        // find bounding box
        let (min_pos, max_pos) = {
            let mut min_pos = start;
            let mut max_pos = start;

            let init = BfsState { pos: start };
            let mut queue = VecDeque::from_iter([init]);

            let mut visited = self.visited.clone();
            let mut iteration = 0;

            while let Some(state) = queue.pop_front() {
                iteration += 1;
                assert!(iteration < MAX_ITER);

                let BfsState { pos: Pos(x, y) } = state;
                let empty = !self.placement.0.get(x, y).expect("within valid area");
                let seen = visited.set(x, y, true).expect("within valid area");
                if !empty || seen {
                    continue;
                }

                min_pos = Pos(x.min(min_pos.0), y.min(min_pos.1));
                max_pos = Pos(x.max(max_pos.0), y.max(max_pos.1));

                if let Some(x) = x.checked_sub(1) {
                    queue.push_back(BfsState { pos: Pos(x, y) });
                }
                if let Some(y) = y.checked_sub(1) {
                    queue.push_back(BfsState { pos: Pos(x, y) });
                }
                if x + 1 < self.placement.0.width {
                    queue.push_back(BfsState { pos: Pos(x + 1, y) });
                }
                if y + 1 < self.placement.0.height {
                    queue.push_back(BfsState { pos: Pos(x, y + 1) });
                }
            }

            (min_pos, max_pos)
        };

        // create connected region
        {
            let width = max_pos.0 - min_pos.0 + 1;
            let height = max_pos.1 - min_pos.1 + 1;
            let occupied = bv::bitvec![1; width * height];
            let mut region = OpenShape(BitShape {
                width,
                height,
                occupied,
            });

            let Pos(min_x, min_y) = min_pos;

            let init = BfsState { pos: start };
            let mut queue = VecDeque::from_iter([init]);

            let mut iteration = 0;

            while let Some(state) = queue.pop_front() {
                iteration += 1;
                assert!(iteration < MAX_ITER);

                let BfsState { pos: Pos(x, y) } = state;
                let empty = !self.placement.0.get(x, y).expect("within valid area");
                let seen = self.visited.set(x, y, true).expect("within valid area");
                if !empty || seen {
                    continue;
                }

                region
                    .set(x - min_x, y - min_y, false)
                    .expect("within bounding box");

                if let Some(x) = x.checked_sub(1) {
                    queue.push_back(BfsState { pos: Pos(x, y) });
                }
                if let Some(y) = y.checked_sub(1) {
                    queue.push_back(BfsState { pos: Pos(x, y) });
                }
                if x + 1 < self.placement.0.width {
                    queue.push_back(BfsState { pos: Pos(x + 1, y) });
                }
                if y + 1 < self.placement.0.height {
                    queue.push_back(BfsState { pos: Pos(x, y + 1) });
                }
            }
            ConnectedPartition(region)
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ConnectedPartition(OpenShape);

impl std::fmt::Display for ConnectedPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ConnectedPartition {
    fn new(width: usize, height: usize) -> Self {
        let occupied = bv::bitvec![0; width * height];
        let shape = BitShape {
            width,
            height,
            occupied,
        };
        Self(OpenShape(shape))
    }

    fn fill_in(&self, shape: &OccupiedShape, pos: Pos) -> Option<Placement> {
        let Pos(x, y) = pos;
        let mut placement = self.0.clone();
        let mut fill_tile = |x, y, b| placement.set(x, y, b).map(|prev| !prev).unwrap_or(false);
        let all_okay = (0..shape.height).all(|j| {
            (0..shape.width).all(|i| {
                let bit = shape[(i, j)];
                match bit {
                    true => fill_tile(x + i, y + j, true),
                    false => true,
                }
            })
        });
        all_okay.then_some(Placement(placement))
    }

    fn iter_shape_placements(&self, shape: &Shape) -> OpenShapePlacementIterator {
        OpenShapePlacementIterator::new(self, shape)
    }
}

impl Iterator for PlacementConnectedPartitionIterator {
    type Item = ConnectedPartition;

    fn next(&mut self) -> Option<Self::Item> {
        let Pos(mut x, y) = match self.last_pos {
            Some(Pos(x, y)) => Pos(x + 1, y),
            None => Pos(0, 0),
        };
        for y in y..self.placement.0.height {
            for x in x..self.placement.0.width {
                let empty = !self.placement.0.get(x, y).expect("within valid area");
                let seen = self.visited.get(x, y).expect("within valid area");
                if empty && !seen {
                    let start = Pos(x, y);
                    let region = self.bfs_connected_partition(start);
                    return Some(region);
                }
            }
            x = 0;
        }
        None
    }
}

#[derive(Debug, Clone)]
struct OccupiedShape {
    shape: BitShape,
    orientation: Orientation,
}

impl std::ops::Deref for OccupiedShape {
    type Target = BitShape;
    fn deref(&self) -> &Self::Target {
        &self.shape
    }
}
impl std::ops::DerefMut for OccupiedShape {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.shape
    }
}

impl OccupiedShape {
    fn from_shape(shape: &Shape) -> Self {
        let grid = &shape.grid;

        let width = grid.width();
        let height = grid.height();
        let mut occupied = bv::BitVec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let tile = grid.get(x, y).expect("within grid");
                let bit = match tile {
                    Tile::Empty => false,
                    Tile::Filled => true,
                };
                occupied.push(bit);
            }
        }
        assert_eq!(occupied.len(), width * height);

        let shape = BitShape {
            width,
            height,
            occupied,
        };
        let orientation = Orientation::Rot0 { flipped: false };
        Self { shape, orientation }
    }

    fn next_orientations(self) -> Self {
        use Orientation::*;

        let OccupiedShape { shape, orientation } = self;
        let (shape, orientation) = match orientation {
            Rot0 { flipped } => (shape.rotate(), Rot90 { flipped }),
            Rot90 { flipped } => (shape.rotate(), Rot180 { flipped }),
            Rot180 { flipped } => (shape.rotate(), Rot270 { flipped }),
            Rot270 { flipped } => (shape.rotate().flip(), Rot0 { flipped: !flipped }),
        };
        Self { shape, orientation }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Side {
    Top,
    Bottom,
    Left,
    Right,
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Orientation {
    Rot0 { flipped: bool },
    Rot90 { flipped: bool },
    Rot180 { flipped: bool },
    Rot270 { flipped: bool },
}

impl Default for Orientation {
    fn default() -> Self {
        Self::Rot0 { flipped: false }
    }
}

#[derive(Debug, Clone)]
struct OpenShapePlacementIteratorRunning {
    connected: ConnectedPartition,
    shape: OccupiedShape,
    progress: (Side, usize),
}

#[derive(Debug, Clone)]
enum OpenShapePlacementIterator {
    Init {
        connected: ConnectedPartition,
        shape: OccupiedShape,
    },
    Running(OpenShapePlacementIteratorRunning),
    Exhausted,
}

#[derive(Debug, Clone)]
struct Placement(OpenShape);

impl std::fmt::Display for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Placement {
    fn iter_connected_partitions(self) -> PlacementConnectedPartitionIterator {
        PlacementConnectedPartitionIterator::new(self)
    }
}

impl Iterator for OpenShapePlacementIterator {
    type Item = Placement;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.move_along_border();
            let Self::Running(running) = &self else {
                return None;
            };
            let Some(pos) = running.search_fitting_pos() else {
                continue;
            };
            let filled = running
                .connected
                .fill_in(&running.shape, pos)
                .expect("verified above");
            return Some(filled);
        }
    }
}

impl OpenShapePlacementIterator {
    fn new(connected: &ConnectedPartition, shape: &Shape) -> Self {
        let connected = connected.clone();
        let shape = OccupiedShape::from_shape(shape);
        OpenShapePlacementIterator::Init { connected, shape }
    }

    fn move_along_border(&mut self) {
        use OpenShapePlacementIterator::*;

        let prev = std::mem::replace(self, Exhausted);
        *self = match prev {
            Init { connected, shape } => {
                let shape_too_big =
                    connected.0.width < shape.width || connected.0.height < shape.height;
                match shape_too_big {
                    true => Exhausted,
                    false => Running(OpenShapePlacementIteratorRunning {
                        connected,
                        shape,
                        progress: (Side::Top, 0),
                    }),
                }
            }
            Running(inner) => {
                let Some(inner) = inner.move_along_border() else {
                    return;
                };
                Running(inner)
            }
            Exhausted => Exhausted,
        };
    }
}

impl OpenShapePlacementIteratorRunning {
    fn move_along_border(self) -> Option<Self> {
        let OpenShapePlacementIteratorRunning {
            connected,
            mut shape,
            progress,
        } = self;

        let (side, along) = progress;
        let end = match side {
            Side::Top | Side::Bottom => connected.0.width - shape.width,
            Side::Left | Side::Right => connected.0.height - shape.height,
        };
        let switch_side = along >= end;
        let progress = match (switch_side, side) {
            (false, side) => (side, along + 1),
            (true, Side::Top) => (Side::Bottom, 0),
            (true, Side::Bottom) => (Side::Left, 0),
            (true, Side::Left) => (Side::Right, 0),
            (true, Side::Right) => {
                shape = shape.next_orientations();
                if shape.orientation == Orientation::default() {
                    return None;
                };
                (Side::Top, 0)
            }
        };
        Some(Self {
            connected,
            shape,
            progress,
        })
    }

    fn search_fitting_pos(&self) -> Option<Pos> {
        let (side, along) = self.progress;

        let candidate_width = self.connected.0.width - self.shape.width + 1;
        let candidate_height = self.connected.0.height - self.shape.height + 1;
        match side {
            Side::Top | Side::Bottom => assert!(along < candidate_width),
            Side::Left | Side::Right => assert!(along < candidate_height),
        }

        match side {
            Side::Top => {
                let x = along;
                (0..candidate_height)
                    .map(|y| Pos(x, y))
                    .find(|&p| self.shape_fits_at(p))
            }
            Side::Bottom => {
                let x = along;
                (0..candidate_height)
                    .rev()
                    .map(|y| Pos(x, y))
                    .find(|&p| self.shape_fits_at(p))
            }
            Side::Left => {
                let y = along;
                (0..candidate_width)
                    .map(|x| Pos(x, y))
                    .find(|&p| self.shape_fits_at(p))
            }
            Side::Right => {
                let y = along;
                (0..candidate_width)
                    .rev()
                    .map(|x| Pos(x, y))
                    .find(|&p| self.shape_fits_at(p))
            }
        }
    }

    fn shape_fits_at(&self, pos: Pos) -> bool {
        let Pos(x, y) = pos;
        let tile_fits_at = |i, j| {
            let region = self.connected.0[(x + i, y + j)];
            let shape = self.shape[(i, j)];
            let overlapping = region && shape;
            !overlapping
        };
        (0..self.shape.height).all(|y| (0..self.shape.width).all(|x| tile_fits_at(x, y)))
    }
}

struct Cache {
    inner: HashMap<(ConnectedPartition, ShapeCounts), bool>,
}

struct Puzzle {
    shapes: Vec<Shape>,
    regions: Vec<Region>,
}

#[derive(Debug, Error)]
pub enum PuzzleParseError {
    #[error("expected shape blocks and a region block separated by two new-lines, got {0}")]
    Block(String),
    #[error("could not parse shape {0}")]
    Shape(usize, #[source] ShapeParseError),
    #[error("could not parse region {0}")]
    Region(usize, #[source] RegionParseError),
}

impl Puzzle {
    fn parse(input: &str) -> Result<Self, PuzzleParseError> {
        let mut blocks = input.split("\n\n");

        let hint = blocks.size_hint();
        let cap = hint.1.unwrap_or(hint.0);
        let mut shapes = Vec::with_capacity(cap);

        let mut last = None;
        while let Some(next) = blocks.next() {
            if let Some(last) = last {
                let i = shapes.len();
                let shape = Shape::parse(last).map_err(|e| PuzzleParseError::Shape(i, e))?;
                shapes.push(shape);
            }

            last = Some(next);
        }
        let Some(last) = last else {
            return Err(PuzzleParseError::Block(input.to_string()));
        };

        let n_shapes = shapes.len();
        let regions = last
            .lines()
            .enumerate()
            .map(|(i, line)| {
                Region::parse(line, n_shapes).map_err(|e| PuzzleParseError::Region(i, e))
            })
            .collect::<Result<_, _>>()?;

        Ok(Self { shapes, regions })
    }

    fn count_fitting_regions(&self) -> usize {
        // let mut cache = Cache {
        //     inner: HashMap::new(),
        // };
        //
        // self.regions
        //     .iter()
        //     .filter(|region| region.shapes_fit(&self.shapes, &mut cache))
        //     .count()
        self.regions
            .iter()
            .filter(|region| region.shapes_fit_naive(&self.shapes))
            .count()
    }
}

#[derive(Debug, Error)]
pub enum Day12Error {
    #[error("error when reading file")]
    ReadError(#[from] std::io::Error),
    #[error("could not parse input")]
    Parsing(#[from] PuzzleParseError),
}

pub fn day12() -> Result<(), Day12Error> {
    let path = std::path::PathBuf::from("resources/day12.txt");
    let mut file = std::fs::File::open(path)?;
    let mut string = String::new();
    file.read_to_string(&mut string)?;
    let puzzle = Puzzle::parse(&string)?;
    let count = puzzle.count_fitting_regions();
    println!("day12: we can place the given shapes into {count} regions");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    static EXAMPLE: &str = "0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2
";

    #[test]
    fn test_input_01() -> anyhow::Result<()> {
        let puzzle = Puzzle::parse(EXAMPLE)?;
        let count = puzzle.count_fitting_regions();
        assert_eq!(count, 2);
        Ok(())
    }
}

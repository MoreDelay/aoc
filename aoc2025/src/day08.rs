use std::{collections::BTreeSet, num::ParseFloatError};

use static_assertions::const_assert;
use thiserror::Error;

#[derive(Debug, Copy, Clone, PartialEq)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Debug, Error)]
pub enum PointParseError {
    #[error("expect format '<number>,<number>,<number>', got '{0}'")]
    WrongFormat(String),
    #[error("could not parse as a number")]
    NotANumber(#[from] ParseFloatError),
}

impl Point {
    fn parse(input: &str) -> Result<Self, PointParseError> {
        let values = input.splitn(4, ',').collect::<Vec<_>>();
        if values.len() != 3 {
            return Err(PointParseError::WrongFormat(input.to_string()));
        }

        let values = values
            .into_iter()
            .map(|v| v.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        let [x, y, z] = values[..] else {
            unreachable!()
        };
        Ok(Self { x, y, z })
    }

    fn get(&self, axis: Axis) -> f64 {
        match axis {
            Axis::X => self.x,
            Axis::Y => self.y,
            Axis::Z => self.z,
        }
    }

    fn get_mut(&mut self, axis: Axis) -> &mut f64 {
        match axis {
            Axis::X => &mut self.x,
            Axis::Y => &mut self.y,
            Axis::Z => &mut self.z,
        }
    }

    fn dist(&self, other: &Point) -> f64 {
        let x = (self.x - other.x).powi(2);
        let y = (self.y - other.y).powi(2);
        let z = (self.z - other.z).powi(2);
        let dist = (x + y + z).sqrt();
        assert!(!dist.is_nan());
        dist
    }
}

#[derive(Debug, Clone, Copy)]
struct AaBb {
    hi: Point,
    lo: Point,
}

impl AaBb {
    fn clamp(&self, p: &Point) -> Point {
        let x = p.x.clamp(self.lo.x, self.hi.x);
        let y = p.y.clamp(self.lo.y, self.hi.y);
        let z = p.z.clamp(self.lo.z, self.hi.z);
        Point { x, y, z }
    }

    fn include(&self, p: &Point) -> Self {
        let hi = Point {
            x: self.hi.x.max(p.x),
            y: self.hi.y.max(p.y),
            z: self.hi.z.max(p.z),
        };
        let lo = Point {
            x: self.lo.x.min(p.x),
            y: self.lo.y.min(p.y),
            z: self.lo.z.min(p.z),
        };
        AaBb { hi, lo }
    }

    fn is_within(&self, p: &Point) -> bool {
        self.clamp(p) == *p
    }

    fn consistent(&self) -> bool {
        (self.lo.x <= self.hi.x) && (self.lo.y <= self.hi.y) && (self.lo.z <= self.hi.z)
    }
}

#[derive(Debug, Copy, Clone)]
enum Axis {
    X,
    Y,
    Z,
}

#[derive(Debug)]
struct Split {
    axis: Axis,
    value: f64,
    lower: usize,
    higher: usize,
}

#[derive(Debug)]
struct Partition {
    aabb: AaBb,
    contained: Vec<usize>,
    split: Option<Split>,
}

struct SpacePartition {
    points: Vec<Point>,
    partitions: Vec<Partition>,
}

impl SpacePartition {
    const SPLIT_THRESHOLD: usize = 5;

    fn new(points: Vec<Point>) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let global_partition = {
            let inf = f64::INFINITY;
            let hi = Point {
                x: -inf,
                y: -inf,
                z: -inf,
            };
            let lo = Point {
                x: inf,
                y: inf,
                z: inf,
            };
            let aabb = AaBb { hi, lo };
            let aabb = points.iter().fold(aabb, |aabb, p| aabb.include(p));

            Partition {
                aabb,
                contained: (0..points.len()).collect(),
                split: None,
            }
        };
        let mut partitions = vec![global_partition];

        struct State {
            parent: usize,
            next_axis: Axis,
        }
        let mut stack = vec![State {
            parent: 0,
            next_axis: Axis::X,
        }];

        while let Some(state) = stack.pop() {
            let State { parent, next_axis } = state;
            let Partition {
                aabb, contained, ..
            } = &partitions[parent];
            let axis = next_axis;

            if contained.len() <= Self::SPLIT_THRESHOLD {
                continue;
            }

            // split space at the median
            let mut contained = contained.clone();
            contained.sort_by(|i, j| {
                let a = points[*i].get(axis);
                let b = points[*j].get(axis);
                a.total_cmp(&b)
            });

            const_assert!(SpacePartition::SPLIT_THRESHOLD >= 2);
            let mid = contained.len() / 2;

            let split_value = {
                let mid0 = contained[mid];
                let mid1 = contained[mid + 1];
                let a = &points[mid0].get(axis);
                let b = &points[mid1].get(axis);
                (a + b) / 2.
            };

            let higher_aabb = {
                let mut aabb = *aabb;
                *aabb.lo.get_mut(axis) = split_value;
                assert!(aabb.consistent());
                aabb
            };
            let lower_aabb = {
                let mut aabb = *aabb;
                *aabb.hi.get_mut(axis) = split_value;
                assert!(aabb.consistent());
                aabb
            };

            let next_axis = match axis {
                Axis::X => Axis::Y,
                Axis::Y => Axis::Z,
                Axis::Z => Axis::X,
            };

            // insert new partitions
            let lower_index = {
                let contained = contained[..=mid].to_vec();
                let partition = Partition {
                    aabb: lower_aabb,
                    contained,
                    split: None,
                };
                assert!(
                    partition
                        .contained
                        .iter()
                        .all(|i| { lower_aabb.is_within(&points[*i]) })
                );

                let index = partitions.len();
                partitions.push(partition);
                stack.push(State {
                    parent: index,
                    next_axis,
                });
                index
            };

            let higher_index = {
                let contained = contained[mid + 1..].to_vec();
                let partition = Partition {
                    aabb: higher_aabb,
                    contained,
                    split: None,
                };
                assert!(
                    partition
                        .contained
                        .iter()
                        .all(|i| higher_aabb.is_within(&points[*i]))
                );
                let index = partitions.len();
                partitions.push(partition);
                stack.push(State {
                    parent: index,
                    next_axis,
                });
                index
            };

            // update child indices in parent
            let split = Split {
                axis,
                value: split_value,
                lower: lower_index,
                higher: higher_index,
            };
            let partition = &mut partitions[parent];
            partition.split = Some(split);
        }

        Some(Self { points, partitions })
    }
}

#[derive(Debug, Clone)]
struct ClosestFound {
    index: usize,
    dist: f64,
}

impl SpacePartition {
    fn initial_descent(
        &self,
        query_index: usize,
        filter: &impl Fn(usize) -> bool,
    ) -> Option<ClosestFound> {
        assert!(query_index < self.points.len());
        if self.points.len() < 2 {
            return None;
        }
        let query = self.points[query_index];

        // go down partition tree
        let leaf_partition = {
            let mut partition_index = 0;

            while let Some(split) = &self.partitions[partition_index].split {
                let cmp_value = match split.axis {
                    Axis::X => query.x,
                    Axis::Y => query.y,
                    Axis::Z => query.z,
                };
                partition_index = match cmp_value < split.value {
                    true => split.lower,
                    false => split.higher,
                };
            }

            partition_index
        };

        self.search_leaf(query_index, leaf_partition, filter)
    }

    fn search_leaf(
        &self,
        query_index: usize,
        leaf_index: usize,
        filter: &impl Fn(usize) -> bool,
    ) -> Option<ClosestFound> {
        assert!(query_index < self.points.len());
        assert!(leaf_index < self.partitions.len());

        let query = self.points[query_index];
        let leaf = &self.partitions[leaf_index];
        assert!(leaf.split.is_none());

        let mut closest_found: Option<ClosestFound> = None;
        for &index in leaf.contained.iter() {
            if !filter(index) {
                continue;
            }
            let p = self.points[index];
            let dist = query.dist(&p);
            if closest_found.as_ref().is_none_or(|q| dist < q.dist) {
                closest_found = Some(ClosestFound { index, dist });
            }
        }

        closest_found
    }

    fn global_search(
        &self,
        query_index: usize,
        filter: &impl Fn(usize) -> bool,
        mut closest_found: ClosestFound,
    ) -> ClosestFound {
        assert!(query_index < self.points.len());
        assert!(closest_found.index < self.points.len());
        assert!(!closest_found.dist.is_nan());

        let query = self.points[query_index];

        struct State {
            index: usize,
        }
        let init = State { index: 0 };
        let mut stack = vec![init];
        while let Some(state) = stack.pop() {
            let State { index } = state;
            let partition = &self.partitions[index];

            // skip this partition if it is impossible is has closer points
            let min_point = partition.aabb.clamp(&query);
            let min_dist = min_point.dist(&query);
            if min_dist >= closest_found.dist {
                continue;
            }

            // descend until leaf
            if let Some(split) = &partition.split {
                let lower = State { index: split.lower };
                let higher = State {
                    index: split.higher,
                };
                stack.push(lower);
                stack.push(higher);
                continue;
            }

            // try to find closest points in this leaf
            if let Some(found) = self.search_leaf(query_index, index, filter)
                && found.dist < closest_found.dist
            {
                closest_found = found;
            }
        }
        closest_found
    }

    /// Finds the closest point (index) into self.points where filter(index) == true. The filter
    /// takes in indices into self.points and returns whether to the given index as search
    /// candidate.
    fn find_closest_filtered(
        &self,
        query_index: usize,
        filter: impl Fn(usize) -> bool,
    ) -> Option<ClosestFound> {
        assert!(query_index < self.points.len());
        let filter = |index| index != query_index && filter(index);
        let cur_closest = self.initial_descent(query_index, &filter)?;
        let cur_closest = self.global_search(query_index, &filter, cur_closest);
        Some(cur_closest)
    }
}

struct GraphForest {
    space_partition: SpacePartition,
    colors: Vec<usize>,
    prev_closest: BTreeSet<(usize, usize)>,
}

impl GraphForest {
    fn new(space_partition: SpacePartition) -> Self {
        let colors = (0..space_partition.points.len()).collect();
        let prev_closest = BTreeSet::new();
        Self {
            space_partition,
            colors,
            prev_closest,
        }
    }

    /// Returns the indices of the two closest points
    fn connect_single(&mut self, filter_for_tree_connections: bool) -> Option<(usize, usize)> {
        struct Candidate {
            from: usize,
            to: usize,
            dist: f64,
        }
        let mut candidate: Option<Candidate> = None;
        for query_index in 0..self.space_partition.points.len() {
            let found = if !filter_for_tree_connections {
                let filter = |index| {
                    let min = query_index.min(index);
                    let max = query_index.max(index);
                    !self.prev_closest.contains(&(min, max))
                };
                self.space_partition
                    .find_closest_filtered(query_index, filter)
            } else {
                let filter = |index| self.colors[query_index] != self.colors[index];
                self.space_partition
                    .find_closest_filtered(query_index, filter)
            };

            let Some(found) = found else { continue };
            if candidate.as_ref().is_none_or(|c| found.dist < c.dist) {
                candidate = Some(Candidate {
                    from: query_index,
                    to: found.index,
                    dist: found.dist,
                })
            }
        }
        let candidate = candidate?;
        let from_color = self.colors[candidate.from];
        let to_color = self.colors[candidate.to];

        if from_color != to_color {
            for color in self.colors.iter_mut() {
                if *color == from_color {
                    *color = to_color;
                }
            }
        }

        let min = candidate.from.min(candidate.to);
        let max = candidate.from.max(candidate.to);
        self.prev_closest.insert((min, max));

        Some((candidate.from, candidate.to))
    }

    fn try_connect_pairwise(&mut self) -> Option<(usize, usize)> {
        self.connect_single(false)
    }

    fn connect_new_wire(&mut self) -> Option<(usize, usize)> {
        self.connect_single(true)
    }

    fn compute_network_product(&self) -> usize {
        let mut colors = self.colors.clone();
        colors.sort();
        let mut largest = [1; 3];

        let mut current_min = largest.iter_mut().min().unwrap();
        let mut color = None;
        let mut count = 1;
        for c in colors {
            if color.is_none_or(|color| color != c) {
                if *current_min < count {
                    *current_min = count;
                    current_min = largest.iter_mut().min().unwrap();
                }

                count = 0;
                color = Some(c);
            }
            count += 1;
        }
        if *current_min < count {
            *current_min = count;
        }
        largest.iter().product()
    }

    fn compute_connection_product(&self, connection: (usize, usize)) -> f64 {
        let (from, to) = connection;
        let from = self.space_partition.points[from];
        let to = self.space_partition.points[to];
        from.x * to.x
    }
}

struct JunctionBoxes(Vec<Point>);

#[derive(Debug, Error)]
pub enum JunctionBoxesParseError {
    #[error("error while reading")]
    ReadError(#[from] std::io::Error),
    #[error("could not parse line {0} as a junction box")]
    PointParseError(usize, #[source] PointParseError),
    #[error("could not parse any point")]
    NoPoints,
}

impl JunctionBoxes {
    fn parse(input: impl std::io::BufRead) -> Result<Self, JunctionBoxesParseError> {
        let boxes = input
            .lines()
            .enumerate()
            .map(|(i, line)| {
                Point::parse(&line?).map_err(|e| JunctionBoxesParseError::PointParseError(i, e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if boxes.is_empty() {
            return Err(JunctionBoxesParseError::NoPoints);
        }
        Ok(Self(boxes))
    }

    fn forest(self) -> GraphForest {
        GraphForest::new(SpacePartition::new(self.0).expect("got enough points by construction"))
    }
}

#[derive(Debug, Error)]
pub enum Day08Error {
    #[error("could not open file")]
    FileError(#[from] std::io::Error),
    #[error("could not parse input")]
    ParseError(#[from] JunctionBoxesParseError),
    #[error("could not connect network the required amount ({1} times), only achieved {0} times")]
    NotEnoughConnections(usize, usize),
}

pub fn day08() -> Result<(), Day08Error> {
    let path = std::path::PathBuf::from("resources/day08.txt");
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let boxes = JunctionBoxes::parse(reader)?;
    let mut forest = boxes.forest();

    let mut last = None;
    for i in 0..1000 {
        last = forest.try_connect_pairwise();
        if last.is_none() {
            return Err(Day08Error::NotEnoughConnections(i, 1000));
        }
    }
    let product = forest.compute_network_product();
    println!("day08: network product is {product}");

    while let Some(now) = forest.connect_new_wire() {
        last = Some(now);
    }
    let product = forest.compute_connection_product(last.unwrap());
    println!("day08: final connection product is {product}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    static EXAMPLE: &str = "162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689";

    #[test]
    fn test_input_01() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let boxes = JunctionBoxes::parse(cursor).unwrap();
        let mut forest = boxes.forest();
        for _ in 0..10 {
            forest.try_connect_pairwise().unwrap();
        }
        let count = forest.compute_network_product();
        assert_eq!(count, 40);
        Ok(())
    }

    #[test]
    fn test_input_02() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let boxes = JunctionBoxes::parse(cursor).unwrap();
        let mut forest = boxes.forest();
        let mut last = None;
        while let Some(now) = forest.connect_new_wire() {
            last = Some(now);
        }
        let last = last.unwrap();
        let result = forest.compute_connection_product(last);
        assert_eq!(result, 25272.);
        Ok(())
    }

    #[ignore]
    #[test]
    fn test_closest_point_query() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let boxes = JunctionBoxes::parse(cursor)?;
        let mut points = boxes.0;
        let expected = points.len();
        let closest = Point {
            x: 900.,
            y: 355.,
            z: 555.,
        };
        points.push(closest);
        let space_partition = SpacePartition::new(points).unwrap();
        let query_index = 2;
        let filter = |index| index != query_index;
        let found = space_partition
            .find_closest_filtered(query_index, filter)
            .unwrap();
        assert_eq!(found.index, expected);
        Ok(())
    }
}

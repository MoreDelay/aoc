use anyhow::Result;
use std::{
    collections::{HashSet, VecDeque},
    fmt::Write,
    marker::PhantomData,
    path::PathBuf,
};

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct Pos(usize, usize);

#[derive(Copy, Clone, Debug, Default)]
struct Visited {
    north: Option<usize>,
    east: Option<usize>,
    south: Option<usize>,
    west: Option<usize>,
}

#[derive(Copy, Clone, Debug)]
enum Tile {
    Empty(Visited),
    Wall,
    Start(Visited),
    End(Visited),
}

impl Tile {
    fn visited(self, dir: Direction) -> Option<usize> {
        let visited = match self {
            Tile::Empty(visited) => visited,
            Tile::Wall => return None,
            Tile::Start(visited) => visited,
            Tile::End(visited) => visited,
        };
        match dir {
            Direction::North => visited.north,
            Direction::East => visited.east,
            Direction::South => visited.south,
            Direction::West => visited.west,
        }
    }

    fn visit(&mut self, dir: Direction, index: usize) {
        let visited = match self {
            Tile::Empty(visited) => visited,
            Tile::Wall => return,
            Tile::Start(visited) => visited,
            Tile::End(visited) => visited,
        };
        match dir {
            Direction::North => visited.north = Some(index),
            Direction::East => visited.east = Some(index),
            Direction::South => visited.south = Some(index),
            Direction::West => visited.west = Some(index),
        };
    }
}

impl Eq for Tile {}

impl PartialEq for Tile {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Tile::Empty(..), Tile::Empty(..)) => true,
            (Tile::Wall, Tile::Wall) => true,
            (Tile::Start(..), Tile::Start(..)) => true,
            (Tile::End(..), Tile::End(..)) => true,
            _ => false,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    fn turn_right(self) -> Self {
        match self {
            Self::North => Self::East,
            Self::East => Self::South,
            Self::South => Self::West,
            Self::West => Self::North,
        }
    }

    fn turn_left(self) -> Self {
        match self {
            Self::North => Self::West,
            Self::East => Self::North,
            Self::South => Self::East,
            Self::West => Self::South,
        }
    }
}

impl std::fmt::Display for Tile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tile::Empty(..) => f.write_char('.'),
            Tile::Wall => f.write_char('#'),
            Tile::Start(..) => f.write_char('S'),
            Tile::End(..) => f.write_char('E'),
        }
    }
}

#[derive(Clone)]
struct Maze {
    size: Pos,
    start: Pos,
    tiles: Vec<Vec<Tile>>,
}

impl Maze {
    fn parse(input: &str) -> Result<Maze> {
        let mut start = None;
        let mut end = None;
        let parser = |x, y, c| match c {
            '.' => Ok(Tile::Empty(Visited::default())),
            '#' => Ok(Tile::Wall),
            'S' if start.is_none() => {
                start = Some(Pos(x, y));
                Ok(Tile::Start(Visited::default()))
            }
            'E' if end.is_none() => {
                end = Some(Pos(x, y));
                Ok(Tile::End(Visited::default()))
            }
            _ => Err(AocError::ParseError),
        };

        let tiles = util::parse_tiles(input, parser)?;
        let start = start.ok_or(AocError::ParseError)?;
        let _end = end.ok_or(AocError::ParseError)?;

        let height = tiles.len();
        let width = tiles[0].len();
        let size = Pos(width, height);

        Ok(Maze { size, start, tiles })
    }

    fn at(&self, pos: Pos) -> Tile {
        let Pos(x, y) = pos;
        self.tiles[y][x]
    }

    fn at_mut(&mut self, pos: Pos) -> &mut Tile {
        let Pos(x, y) = pos;
        &mut self.tiles[y][x]
    }

    fn neighbor_pos(&self, pos: Pos, dir: Direction) -> Option<Pos> {
        let Pos(width, height) = self.size;
        let Pos(x, y) = pos;
        match dir {
            Direction::North => (y > 0).then(|| Pos(x, y - 1)),
            Direction::East => (x < width - 1).then(|| Pos(x + 1, y)),
            Direction::South => (y < height - 1).then(|| Pos(x, y + 1)),
            Direction::West => (x > 0).then(|| Pos(x - 1, y)),
        }
    }
}

#[derive(Clone, Debug)]
struct MoveNode {
    pos: Pos,
    facing: Direction,
    points: usize,
    last: Vec<usize>,
}

struct Init;
struct Done;

trait DijkstraState {}
impl DijkstraState for Init {}
impl DijkstraState for Done {}

struct Dijkstra<T: DijkstraState> {
    maze: Maze,
    nodes: Vec<MoveNode>,
    unvisited: Vec<usize>,
    best_points: Option<usize>,
    best_paths: Vec<MoveNode>,
    state: PhantomData<T>,
}

impl Dijkstra<Init> {
    fn new(maze: Maze) -> Self {
        let initial = MoveNode {
            pos: maze.start,
            facing: Direction::East,
            points: 0,
            last: Vec::new(),
        };
        let nodes = vec![initial];
        let unvisited = vec![0];
        Self {
            maze,
            nodes,
            unvisited,
            best_points: None,
            best_paths: Vec::new(),
            state: PhantomData,
        }
    }

    fn get_next_unvisited(&mut self) -> Option<(usize, MoveNode)> {
        if self.unvisited.is_empty() {
            return None;
        }

        let mut smallest_index = None;
        let mut smallest_points = None;
        for (ii, i) in self.unvisited.iter().enumerate() {
            let cur_points = self.nodes[*i].points;
            if smallest_points.is_none_or(|p| p > cur_points) {
                smallest_points = Some(cur_points);
                smallest_index = Some(ii);
            }
        }

        smallest_index.map(|ii| {
            let index = self.unvisited.remove(ii);
            let node = self.nodes[index].clone();
            (index, node)
        })
    }

    fn solve(mut self) -> Dijkstra<Done> {
        while let Some((cur_index, node)) = self.get_next_unvisited() {
            let MoveNode {
                pos,
                facing,
                points,
                ..
            } = node;

            if self.best_points.is_some_and(|p| p < points) {
                // all further paths will have worse score
                break;
            }

            let current_tile = self.maze.at(pos);
            if let Tile::End(_) = current_tile {
                assert!(self.best_points.is_none_or(|p| p == points));
                self.best_points = Some(points);
                self.best_paths.push(node);
                continue;
            }

            // look around (assume out of bounds is wall)
            let left = self.maze.neighbor_pos(pos, facing.turn_left());
            let left = left.map(|p| self.maze.at(p)).unwrap_or(Tile::Wall);
            let right = self.maze.neighbor_pos(pos, facing.turn_right());
            let right = right.map(|p| self.maze.at(p)).unwrap_or(Tile::Wall);

            // add new node for turning left
            let facing_left = facing.turn_left();
            if left != Tile::Wall {
                if let Some(prev_index) = current_tile.visited(facing_left) {
                    let prev_node = &mut self.nodes[prev_index];
                    if prev_node.points == points + 1000 {
                        prev_node.last.push(cur_index);
                    }
                } else {
                    let new_index = self.nodes.len();
                    let new_node = MoveNode {
                        pos,
                        facing: facing_left,
                        points: points + 1000,
                        last: vec![cur_index],
                    };
                    self.unvisited.push(new_index);
                    self.nodes.push(new_node);

                    self.maze.at_mut(pos).visit(facing_left, new_index);
                }
            }

            // add new node for turning right
            let facing_right = facing.turn_right();
            if right != Tile::Wall {
                if let Some(prev_index) = current_tile.visited(facing_right) {
                    let prev_node = &mut self.nodes[prev_index];
                    if prev_node.points == points + 1000 {
                        prev_node.last.push(cur_index);
                    }
                } else {
                    let new_index = self.nodes.len();
                    let new_node = MoveNode {
                        pos,
                        facing: facing_right,
                        points: points + 1000,
                        last: vec![cur_index],
                    };
                    self.unvisited.push(new_index);
                    self.nodes.push(new_node);

                    self.maze.at_mut(pos).visit(facing_right, new_index);
                }
            }

            let Some(new_pos) = self.maze.neighbor_pos(pos, facing) else {
                continue;
            };

            // add new node for moving forward
            let tile_ahead = self.maze.at(new_pos);
            if tile_ahead != Tile::Wall {
                if let Some(prev_index) = tile_ahead.visited(facing) {
                    let prev_node = &mut self.nodes[prev_index];
                    if prev_node.points == points + 1 {
                        prev_node.last.push(cur_index);
                    }
                } else {
                    let new_index = self.nodes.len();
                    let new_node = MoveNode {
                        pos: new_pos,
                        facing,
                        points: points + 1,
                        last: vec![cur_index],
                    };
                    self.unvisited.push(new_index);
                    self.nodes.push(new_node);

                    self.maze.at_mut(pos).visit(facing, new_index);
                }
            }
        }

        let Dijkstra {
            maze,
            nodes,
            unvisited,
            best_points,
            best_paths,
            ..
        } = self;
        Dijkstra {
            maze,
            nodes,
            unvisited,
            best_points,
            best_paths,
            state: PhantomData,
        }
    }
}

impl Dijkstra<Done> {
    fn best_points(&self) -> Option<usize> {
        self.best_points
    }

    fn best_paths(&self) -> (&Vec<MoveNode>, &Vec<MoveNode>) {
        (&self.best_paths, &self.nodes)
    }
}

fn count_unique_pos_and_dir(
    final_nodes: &[MoveNode],
    nodes: &[MoveNode],
) -> HashSet<(Pos, Direction)> {
    let mut set = HashSet::new();
    for final_node in final_nodes {
        let MoveNode {
            pos, facing, last, ..
        } = final_node;
        set.insert((*pos, *facing));
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.extend(last);
        while let Some(last) = queue.pop_front() {
            let node = &nodes[last];
            let MoveNode {
                pos, facing, last, ..
            } = node;

            if set.contains(&(*pos, *facing)) {
                continue;
            }
            set.insert((*pos, *facing));
            queue.extend(last);
        }
    }
    set
}

fn count_seats(final_nodes: &[MoveNode], nodes: &[MoveNode]) -> usize {
    let set = count_unique_pos_and_dir(final_nodes, nodes);
    let set = set.iter().map(|&(p, _)| p).collect::<HashSet<_>>();
    set.len()
}

pub fn run() -> Result<()> {
    println!("day 16");
    let path = PathBuf::from("./resources/day16.txt");
    let data = util::get_data_string(&path)?;
    let maze = Maze::parse(&data)?;
    let dijkstra = Dijkstra::new(maze);
    let dijkstra = dijkstra.solve();
    let best_points = dijkstra
        .best_points()
        .ok_or(AocError::ValueError("could not solve maze".into()))?;
    println!("least points to solve maze: {best_points}");

    let (final_nodes, all_nodes) = dijkstra.best_paths();
    let count = count_seats(final_nodes, all_nodes);
    println!("number of seats: {count}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_maze() {
        let input = "###############
#.......#....E#
#.#.###.#.###.#
#.....#.#...#.#
#.###.#####.#.#
#.#.#.......#.#
#.#.#####.###.#
#...........#.#
###.#.#####.#.#
#...#.....#.#.#
#.#.#.###.#.#.#
#.....#...#.#.#
#.###.#.#.#.#.#
#S..#.....#...#
###############";
        let maze = Maze::parse(input).unwrap();
        let dijkstra = Dijkstra::new(maze);
        let dijkstra = dijkstra.solve();
        let best_points = dijkstra.best_points().unwrap();
        assert_eq!(best_points, 7036);
    }

    #[test]
    fn test_maze_path_unique_pos_and_dir() {
        let input = "###############
#.......#....E#
#.#.###.#.###.#
#.....#.#...#.#
#.###.#####.#.#
#.#.#.......#.#
#.#.#####.###.#
#...........#.#
###.#.#####.#.#
#...#.....#.#.#
#.#.#.###.#.#.#
#.....#...#.#.#
#.###.#.#.#.#.#
#S..#.....#...#
###############";
        let maze = Maze::parse(input).unwrap();
        let dijkstra = Dijkstra::new(maze);
        let dijkstra = dijkstra.solve();
        let (final_nodes, all_nodes) = dijkstra.best_paths();
        let set = count_unique_pos_and_dir(final_nodes, all_nodes);
        assert!(set.contains(&(Pos(1, 13), Direction::East)));
        assert!(set.contains(&(Pos(1, 13), Direction::North)));
        assert!(set.contains(&(Pos(1, 10), Direction::North)));
        assert!(set.contains(&(Pos(2, 11), Direction::East)));
        assert!(set.contains(&(Pos(3, 10), Direction::North)));
        assert!(set.contains(&(Pos(3, 8), Direction::North)));
        assert!(set.contains(&(Pos(4, 11), Direction::East)));
        assert!(set.contains(&(Pos(6, 7), Direction::East)));
    }

    #[test]
    fn test_maze_seats() {
        let input = "###############
#.......#....E#
#.#.###.#.###.#
#.....#.#...#.#
#.###.#####.#.#
#.#.#.......#.#
#.#.#####.###.#
#...........#.#
###.#.#####.#.#
#...#.....#.#.#
#.#.#.###.#.#.#
#.....#...#.#.#
#.###.#.#.#.#.#
#S..#.....#...#
###############";
        let maze = Maze::parse(input).unwrap();
        let dijkstra = Dijkstra::new(maze);
        let dijkstra = dijkstra.solve();
        let (final_nodes, all_nodes) = dijkstra.best_paths();
        let count = count_seats(final_nodes, all_nodes);

        assert_eq!(count, 45);
    }
}

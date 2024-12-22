use anyhow::Result;
use std::{fmt::Write, path::PathBuf};

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug)]
struct Pos(usize, usize);

#[derive(Copy, Clone, Debug, Default)]
struct Visited {
    north: bool,
    east: bool,
    south: bool,
    west: bool,
}

#[derive(Copy, Clone, Debug)]
enum Tile {
    Empty(Visited),
    Wall,
    Start(Visited),
    End(Visited),
}

impl Tile {
    fn visited(self, dir: Direction) -> bool {
        let visited = match self {
            Tile::Empty(visited) => visited,
            Tile::Wall => return false,
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

    fn visit(&mut self, dir: Direction) {
        let visited = match self {
            Tile::Empty(visited) => visited,
            Tile::Wall => return,
            Tile::Start(visited) => visited,
            Tile::End(visited) => visited,
        };
        match dir {
            Direction::North => visited.north = true,
            Direction::East => visited.east = true,
            Direction::South => visited.south = true,
            Direction::West => visited.west = true,
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

#[derive(Copy, Clone, Debug)]
struct MoveNode {
    pos: Pos,
    facing: Direction,
    points: usize,
    #[allow(dead_code)]
    last: Option<usize>,
}

struct Dijkstra<'a> {
    maze: &'a mut Maze,
    nodes: Vec<MoveNode>,
    unvisited: Vec<usize>,
    best: Option<MoveNode>,
}

impl<'a> Dijkstra<'a> {
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
            let node = self.nodes[index];
            (index, node)
        })
    }

    fn search_goal(&mut self) -> Option<MoveNode> {
        while let Some((index, node)) = self.get_next_unvisited() {
            let MoveNode {
                pos,
                facing,
                points,
                ..
            } = node;

            // look around (assume out of bounds is wall)
            let current_tile = self.maze.at(pos);
            let left = self.maze.neighbor_pos(pos, facing.turn_left());
            let left = left.map(|p| self.maze.at(p)).unwrap_or(Tile::Wall);
            let right = self.maze.neighbor_pos(pos, facing.turn_right());
            let right = right.map(|p| self.maze.at(p)).unwrap_or(Tile::Wall);

            if let Tile::End(_) = current_tile {
                self.best = Some(node);
                return Some(node);
            }

            // add new node for turning left
            let turn_left = facing.turn_left();
            if left != Tile::Wall && !current_tile.visited(turn_left) {
                self.maze.at_mut(pos).visit(turn_left);

                let new_index = self.nodes.len();
                let new_node = MoveNode {
                    pos,
                    facing: turn_left,
                    points: points + 1000,
                    last: Some(index),
                };
                self.unvisited.push(new_index);
                self.nodes.push(new_node);
            }

            // add new node for turning right
            let turn_right = facing.turn_right();
            if right != Tile::Wall && !current_tile.visited(turn_right) {
                self.maze.at_mut(pos).visit(turn_right);

                let new_index = self.nodes.len();
                let new_node = MoveNode {
                    pos,
                    facing: turn_right,
                    points: points + 1000,
                    last: Some(index),
                };
                self.unvisited.push(new_index);
                self.nodes.push(new_node);
            }

            let Some(new_pos) = self.maze.neighbor_pos(pos, facing) else {
                continue;
            };

            // add new node for moving forward
            let tile_ahead = self.maze.at(new_pos);
            if tile_ahead != Tile::Wall && !tile_ahead.visited(facing) {
                self.maze.at_mut(pos).visit(facing);

                let new_index = self.nodes.len();
                let new_node = MoveNode {
                    pos: new_pos,
                    facing,
                    points: points + 1,
                    last: Some(index),
                };
                self.unvisited.push(new_index);
                self.nodes.push(new_node);
            }
        }
        None
    }
}

fn solve_maze(maze: &mut Maze) -> Option<usize> {
    let initial = MoveNode {
        pos: maze.start,
        facing: Direction::East,
        points: 0,
        last: None,
    };
    let nodes = vec![initial];
    let unvisited = vec![0];
    let mut dijkstra = Dijkstra {
        maze,
        nodes,
        unvisited,
        best: None,
    };

    let best = dijkstra.search_goal();
    best.map(|n| n.points)
}

pub fn run() -> Result<()> {
    println!("day 16");
    let path = PathBuf::from("./resources/day16.txt");
    let data = util::get_data_string(&path)?;
    let mut maze = Maze::parse(&data)?;
    let best = solve_maze(&mut maze).ok_or(AocError::ValueError("could not solve maze".into()))?;
    println!("least points to solve maze: {best}");
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
        let mut maze = Maze::parse(input).unwrap();
        let best = solve_maze(&mut maze).unwrap();
        assert_eq!(best, 7036);
    }
}

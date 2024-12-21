use anyhow::Result;
use std::{fmt::Write, path::PathBuf};

use crate::util::{self, AocError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Up,
    Right,
    Down,
    Left,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tile {
    Empty,
    Wall,
    SmallBox,
    BigBoxL,
    BigBoxR,
    Robot,
}

impl std::fmt::Display for Tile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tile::Empty => f.write_char('.'),
            Tile::Wall => f.write_char('#'),
            Tile::SmallBox => f.write_char('O'),
            Tile::BigBoxL => f.write_char('['),
            Tile::BigBoxR => f.write_char(']'),
            Tile::Robot => f.write_char('@'),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pos(usize, usize);

struct Warehouse {
    size: Pos,
    robot: Pos,
    tiles: Vec<Vec<Tile>>,
}

impl Warehouse {
    fn at(&self, pos: Pos) -> Tile {
        let Pos(x, y) = pos;
        self.tiles[y][x]
    }

    fn set(&mut self, pos: Pos, tile: Tile) {
        let Pos(x, y) = pos;
        self.tiles[y][x] = tile;
    }

    fn neighbor_pos(&self, pos: Pos, dir: Direction) -> Option<Pos> {
        let Pos(width, height) = self.size;
        let Pos(x, y) = pos;
        match dir {
            Direction::Up => (y > 0).then(|| Pos(x, y - 1)),
            Direction::Right => (x < width - 1).then(|| Pos(x + 1, y)),
            Direction::Down => (y < height - 1).then(|| Pos(x, y + 1)),
            Direction::Left => (x > 0).then(|| Pos(x - 1, y)),
        }
    }

    // returns None if any position is blocked by a wall
    fn find_pushed_boxes(&self, dir: Direction, pushing: &Vec<Pos>) -> Option<Vec<Pos>> {
        // push in ascending order (x-coord) per row
        let mut pushed = Vec::new();
        // keep track of smallest index so that we do not duplicate work later (diamond problem)
        let mut smallest = None;

        for &pos in pushing {
            let Some(neighbor) = self.neighbor_pos(pos, dir) else {
                return None; // can not push box out of bounds
            };
            let tile = self.at(neighbor);
            let (box1, box2) = match (tile, dir) {
                (Tile::Empty, _) => continue,
                (Tile::Wall, _) => return None,
                (Tile::SmallBox, _) => (neighbor, None),
                (Tile::BigBoxL, Direction::Up | Direction::Down) => {
                    let box_l = neighbor;
                    let box_r = self.neighbor_pos(neighbor, Direction::Right).unwrap();
                    (box_l, Some(box_r))
                }
                (Tile::BigBoxR, Direction::Up | Direction::Down) => {
                    let box_l = self.neighbor_pos(neighbor, Direction::Left).unwrap();
                    let box_r = neighbor;
                    (box_l, Some(box_r))
                }
                (Tile::BigBoxL, _) => (neighbor, None),
                (Tile::BigBoxR, _) => (neighbor, None),
                (Tile::Robot, _) => unreachable!(),
            };

            if let Some(smallest) = smallest {
                let Pos(x, _y) = box1;
                if x <= smallest {
                    continue;
                }
            }
            smallest = box2.and_then(|box2| Some(box2.0)).or_else(|| Some(box1.0));

            pushed.push(box1);
            if let Some(box2) = box2 {
                pushed.push(box2);
            }
        }
        Some(pushed)
    }

    fn move_and_push_boxes(&mut self, dir: Direction) -> bool {
        let mut pushed_rows = Vec::new();

        let mut pushing = vec![self.robot];
        pushed_rows.push(pushing.clone());

        while !pushing.is_empty() {
            let Some(pushed) = self.find_pushed_boxes(dir, &pushing) else {
                return false;
            };
            pushed_rows.push(pushed.clone());
            pushing = pushed;
        }

        for row in pushed_rows.iter().rev() {
            for &pos in row {
                let tile = self.at(pos);
                let empty = self.neighbor_pos(pos, dir).unwrap();
                // println!("push {tile} from {pos:?} to {empty:?}");
                self.set(empty, tile);
                self.set(pos, Tile::Empty);
                if let Tile::Robot = tile {
                    self.robot = empty;
                }
            }
        }

        return true;
    }

    fn execute_protocol(&mut self, moves: &Vec<Direction>) {
        // println!("start:\n{self}");
        for &dir in moves.iter() {
            self.move_and_push_boxes(dir);
            // println!("with {dir:?}:\n{self}");
        }
    }

    fn compute_gps_sum(&self) -> usize {
        self.tiles
            .iter()
            .enumerate()
            .map(|(y, row)| {
                row.iter().enumerate().map(move |(x, t)| match t {
                    Tile::SmallBox => y * 100 + x,
                    Tile::BigBoxL => y * 100 + x,
                    _ => 0,
                })
            })
            .flatten()
            .sum()
    }
}

impl std::fmt::Display for Warehouse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Pos(_, height) = self.size;
        for (index, row) in self.tiles.iter().enumerate() {
            for t in row.iter() {
                t.fmt(f)?;
            }

            if index < height - 1 {
                f.write_char('\n')?;
            }
        }
        Ok(())
    }
}

fn parse_small_warehouse(input: &str) -> Result<(Warehouse, Vec<Direction>)> {
    let mut split_iter = input.split("\n\n");
    let Some(tiles) = split_iter.next() else {
        return Err(AocError::ParseError.into());
    };
    let Some(moves) = split_iter.next() else {
        return Err(AocError::ParseError.into());
    };
    if !split_iter.next().is_none() {
        return Err(AocError::ParseError.into());
    };

    let moves = moves
        .chars()
        .filter(|&c| c != '\n')
        .map(|c| match c {
            '^' => Ok(Direction::Up),
            '>' => Ok(Direction::Right),
            'v' => Ok(Direction::Down),
            '<' => Ok(Direction::Left),
            _ => Err(AocError::ParseError),
        })
        .collect::<Result<Vec<_>, AocError>>()?;

    let mut robot_at = None;
    let parser = |x, y, c| match c {
        '.' => Ok(Tile::Empty),
        '#' => Ok(Tile::Wall),
        'O' => Ok(Tile::SmallBox),
        '@' if robot_at.is_none() => {
            robot_at = Some(Pos(x, y));
            Ok(Tile::Robot)
        }
        _ => Err(AocError::ParseError),
    };
    let tiles = util::parse_tiles(tiles, parser)?;

    let Some(robot) = robot_at else {
        return Err(AocError::ParseError.into());
    };
    // when we know where the robot is, we know we have at least one row
    let size = Pos(tiles[0].len(), tiles.len());

    let warehouse = Warehouse { size, robot, tiles };
    Ok((warehouse, moves))
}

fn parse_big_warehouse(input: &str) -> Result<(Warehouse, Vec<Direction>)> {
    let mut split_iter = input.split("\n\n");
    let Some(tiles) = split_iter.next() else {
        return Err(AocError::ParseError.into());
    };
    let Some(moves) = split_iter.next() else {
        return Err(AocError::ParseError.into());
    };
    if !split_iter.next().is_none() {
        return Err(AocError::ParseError.into());
    };

    let moves = moves
        .chars()
        .filter(|&c| c != '\n')
        .map(|c| match c {
            '^' => Ok(Direction::Up),
            '>' => Ok(Direction::Right),
            'v' => Ok(Direction::Down),
            '<' => Ok(Direction::Left),
            _ => Err(AocError::ParseError),
        })
        .collect::<Result<Vec<_>, AocError>>()?;

    // parser creates 2 tiles per character
    let mut robot_at = None;
    let parser = |x, y, c| match c {
        '.' => Ok([Tile::Empty, Tile::Empty]),
        '#' => Ok([Tile::Wall, Tile::Wall]),
        'O' => Ok([Tile::BigBoxL, Tile::BigBoxR]),
        '@' if robot_at.is_none() => {
            robot_at = Some(Pos(x * 2, y));
            Ok([Tile::Robot, Tile::Empty])
        }
        _ => Err(AocError::ParseError),
    };
    let tiles = util::parse_tiles(tiles, parser)?;
    // flatten lowest level
    let tiles: Vec<Vec<_>> = tiles
        .into_iter()
        .map(|r| r.into_iter().flatten().collect())
        .collect();

    let Some(robot) = robot_at else {
        return Err(AocError::ParseError.into());
    };
    // when we know where the robot is, we know we have at least one row
    let size = Pos(tiles[0].len(), tiles.len());

    let warehouse = Warehouse { size, robot, tiles };
    Ok((warehouse, moves))
}

pub fn run() -> Result<()> {
    println!("day 15");
    let path = PathBuf::from("./resources/day15.txt");
    let data = util::get_data_string(&path)?;
    let (mut warehouse, moves) = parse_small_warehouse(&data)?;
    warehouse.execute_protocol(&moves);
    let gps_sum = warehouse.compute_gps_sum();
    println!("GPS sum: {gps_sum}");
    let (mut warehouse, moves) = parse_big_warehouse(&data)?;
    warehouse.execute_protocol(&moves);
    let gps_sum = warehouse.compute_gps_sum();
    println!("GPS sum: {gps_sum}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_small_warehouse() {
        let input = "##########
#..O..O.O#
#......O.#
#.OO..O.O#
#..O@..O.#
#O#..O...#
#O..O..O.#
#.OO.O.OO#
#....O...#
##########

<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^
v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^";
        let (warehouse, moves) = parse_small_warehouse(input).unwrap();
        let Pos(width, height) = warehouse.size;

        assert_eq!(width, 10);
        assert_eq!(height, 10);
        assert_eq!(warehouse.tiles[0].len(), width);
        assert_eq!(warehouse.tiles.len(), height);
        assert_eq!(moves.len(), 140);

        let robot_tile = warehouse.at(warehouse.robot);
        assert_eq!(robot_tile, Tile::Robot);
    }

    #[test]
    fn test_parse_big_warehouse() {
        let input = "##########
#..O..O.O#
#......O.#
#.OO..O.O#
#..O@..O.#
#O#..O...#
#O..O..O.#
#.OO.O.OO#
#....O...#
##########

<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^
v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^";
        let (warehouse, moves) = parse_big_warehouse(input).unwrap();
        let Pos(width, height) = warehouse.size;

        assert_eq!(width, 20);
        assert_eq!(height, 10);
        assert_eq!(warehouse.tiles[0].len(), width);
        assert_eq!(warehouse.tiles.len(), height);
        assert_eq!(moves.len(), 140);

        let robot_tile = warehouse.at(warehouse.robot);
        assert_eq!(robot_tile, Tile::Robot);

        let expected = "####################
##....[]....[]..[]##
##............[]..##
##..[][]....[]..[]##
##....[]@.....[]..##
##[]##....[]......##
##[]....[]....[]..##
##..[][]..[]..[][]##
##........[]......##
####################";
        let output = format!("{warehouse}");
        assert_eq!(output, expected);
    }

    #[test]
    fn test_example_small() {
        let input = "##########
#..O..O.O#
#......O.#
#.OO..O.O#
#..O@..O.#
#O#..O...#
#O..O..O.#
#.OO.O.OO#
#....O...#
##########

<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^
vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v
><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<
<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^
^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><
^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^
>^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^
<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>
^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>
v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^";
        let (mut warehouse, moves) = parse_small_warehouse(input).unwrap();
        warehouse.execute_protocol(&moves);
        let expected = "##########
#.O.O.OOO#
#........#
#OO......#
#OO@.....#
#O#.....O#
#O.....OO#
#O.....OO#
#OO....OO#
##########";
        let output = format!("{warehouse}");
        assert_eq!(output, expected);
        let gps_sum = warehouse.compute_gps_sum();
        assert_eq!(gps_sum, 10092);
    }

    #[test]
    fn test_example_big() {
        let input = "##########
#..O..O.O#
#......O.#
#.OO..O.O#
#..O@..O.#
#O#..O...#
#O..O..O.#
#.OO.O.OO#
#....O...#
##########

<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^
vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v
><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<
<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^
^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><
^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^
>^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^
<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>
^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>
v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^";
        let (mut warehouse, moves) = parse_big_warehouse(input).unwrap();
        warehouse.execute_protocol(&moves);
        let expected = "####################
##[].......[].[][]##
##[]...........[].##
##[]........[][][]##
##[]......[]....[]##
##..##......[]....##
##..[]............##
##..@......[].[][]##
##......[][]..[]..##
####################";
        let output = format!("{warehouse}");
        assert_eq!(output, expected);
        let gps_sum = warehouse.compute_gps_sum();
        assert_eq!(gps_sum, 9021);
    }
}

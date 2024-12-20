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

#[derive(Clone, Copy, Debug)]
enum Tile {
    Empty,
    Wall,
    Box,
    Robot,
}

impl std::fmt::Display for Tile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tile::Empty => f.write_char('.'),
            Tile::Wall => f.write_char('#'),
            Tile::Box => f.write_char('O'),
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

    fn move_and_push_boxes(&mut self, dir: Direction) -> bool {
        let first_pos = self.neighbor_pos(self.robot, dir);
        let mut current_pos = first_pos;
        loop {
            let Some(pos) = current_pos else { return false };
            match self.at(pos) {
                Tile::Empty => break,
                Tile::Wall => return false,
                Tile::Box => current_pos = self.neighbor_pos(pos, dir),
                Tile::Robot => unreachable!(),
            }
        }
        let Some(last_pos) = current_pos else {
            return false;
        };
        let first_pos = first_pos.unwrap();

        if last_pos != first_pos {
            self.set(last_pos, Tile::Box);
        }

        self.set(self.robot, Tile::Empty);
        self.set(first_pos, Tile::Robot);
        self.robot = first_pos;
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
                    Tile::Box => y * 100 + x,
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

fn parse_warehouse(input: &str) -> Result<(Warehouse, Vec<Direction>)> {
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
        'O' => Ok(Tile::Box),
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

pub fn run() -> Result<()> {
    println!("day 15");
    let path = PathBuf::from("./resources/day15.txt");
    let data = util::get_data_string(&path)?;
    let (mut warehouse, moves) = parse_warehouse(&data)?;
    warehouse.execute_protocol(&moves);
    let gps_sum = warehouse.compute_gps_sum();
    println!("GPS sum: {gps_sum}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_warehouse() {
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
        let (warehouse, moves) = parse_warehouse(input).unwrap();
        let Pos(width, height) = warehouse.size;

        assert_eq!(width, 10);
        assert_eq!(height, 10);
        assert_eq!(warehouse.tiles[0].len(), width);
        assert_eq!(warehouse.tiles.len(), height);
        assert_eq!(moves.len(), 140);
    }

    #[test]
    fn test_example() {
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
        let (mut warehouse, moves) = parse_warehouse(input).unwrap();
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
}

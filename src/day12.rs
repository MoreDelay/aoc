use anyhow::Result;
use std::{collections::VecDeque, path::PathBuf};

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug, Default)]
struct Tile {
    plant: char,
    visited: bool,
    mark_up: bool,
    mark_right: bool,
    mark_down: bool,
    mark_left: bool,
}

impl Tile {
    fn is_marked(&self, dir: Direction) -> bool {
        match dir {
            Direction::Up => self.mark_up,
            Direction::Right => self.mark_right,
            Direction::Down => self.mark_down,
            Direction::Left => self.mark_left,
        }
    }

    fn mark(&mut self, dir: Direction) {
        match dir {
            Direction::Up => self.mark_up = true,
            Direction::Right => self.mark_right = true,
            Direction::Down => self.mark_down = true,
            Direction::Left => self.mark_left = true,
        }
    }
}

#[derive(Clone)]
struct Plot {
    width: usize,
    height: usize,
    data: Vec<Vec<Tile>>,
}

impl Plot {
    fn within(&self, pos: Pos) -> bool {
        let Pos(x, y) = pos;
        x < self.width && y < self.height
    }

    fn get_mut(&mut self, pos: Pos) -> Result<&mut Tile, AocError> {
        if !self.within(pos) {
            return Err(AocError::ValueError("not within plot".to_string()));
        }
        let Pos(x, y) = pos;
        Ok(&mut self.data[y][x])
    }

    fn get(&self, pos: Pos) -> Result<Tile, AocError> {
        if !self.within(pos) {
            return Err(AocError::ValueError("not within plot".to_string()));
        }
        let Pos(x, y) = pos;
        Ok(self.data[y][x])
    }

    fn move_pos(&self, pos: Pos, dir: Direction) -> Option<Pos> {
        let Pos(x, y) = pos;
        match dir {
            Direction::Up => (y > 0).then(|| Pos(x, y - 1)),
            Direction::Right => (x < self.width - 1).then(|| Pos(x + 1, y)),
            Direction::Down => (y < self.height - 1).then(|| Pos(x, y + 1)),
            Direction::Left => (x > 0).then(|| Pos(x - 1, y)),
        }
    }
}

fn parse_plot(input: &str) -> Result<Plot> {
    let parser = |_, _, c| {
        Ok(Tile {
            plant: c,
            ..Default::default()
        })
    };
    let data = util::parse_tiles(input, parser)?;
    let height = data.len();
    let width = if height > 0 { data[0].len() } else { 0 };
    Ok(Plot {
        width,
        height,
        data,
    })
}

fn get_price_at(pos: Pos, plot: &mut Plot) -> usize {
    let expected = match plot.get(pos).expect("start within plot") {
        Tile { visited, .. } if visited => return 0,
        Tile { plant: name, .. } => name,
    };

    let mut perimeter = 0;
    let mut fields = 0;

    let mut queue = VecDeque::new();
    queue.push_back(Some(pos));
    while !queue.is_empty() {
        let next = queue.pop_front().expect("checked not empty");
        let tile = next.map(|p| plot.get_mut(p).expect("push only valid pos into queue"));
        match tile {
            None => {
                perimeter += 1;
                continue;
            }
            Some(Tile { plant: name, .. }) if *name != expected => {
                perimeter += 1;
                continue;
            }
            Some(Tile { visited, .. }) if !*visited => {
                *visited = true;
                fields += 1;
            }
            Some(_) => {
                continue;
            }
        }

        let Pos(x, y) = next.expect("checked in match");
        queue.push_back((x > 0).then(|| Pos(x - 1, y)));
        queue.push_back((y > 0).then(|| Pos(x, y - 1)));
        queue.push_back((x < plot.width - 1).then(|| Pos(x + 1, y)));
        queue.push_back((y < plot.height - 1).then(|| Pos(x, y + 1)));
    }

    perimeter * fields
}

fn get_perimeter_price(mut plot: Plot) -> usize {
    let mut price = 0;
    for y in 0..plot.height {
        for x in 0..plot.width {
            price += get_price_at(Pos(x, y), &mut plot);
        }
    }
    price
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Direction {
    Up,
    Right,
    Down,
    Left,
}

impl Direction {
    const fn rotate(self) -> Self {
        match self {
            Direction::Up => Direction::Right,
            Direction::Right => Direction::Down,
            Direction::Down => Direction::Left,
            Direction::Left => Direction::Up,
        }
    }

    const fn rotate_rev(self) -> Self {
        const FOR_UP: Direction = Direction::Up.rotate().rotate().rotate();
        const FOR_RIGHT: Direction = Direction::Right.rotate().rotate().rotate();
        const FOR_DOWN: Direction = Direction::Down.rotate().rotate().rotate();
        const FOR_LEFT: Direction = Direction::Left.rotate().rotate().rotate();

        match self {
            Direction::Up => FOR_UP,
            Direction::Right => FOR_RIGHT,
            Direction::Down => FOR_DOWN,
            Direction::Left => FOR_LEFT,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct BorderState {
    expected: char,
    pos: Pos,
    outside: Direction,
}

fn get_next_state(current_state: BorderState, plot: &Plot) -> BorderState {
    let BorderState {
        expected,
        pos,
        outside,
    } = current_state;

    // check for inner curve
    let in_front = plot.move_pos(pos, outside);
    if let Some(new_pos) = in_front {
        let plant = plot.get(new_pos).expect("checked").plant;
        if plant == expected {
            return BorderState {
                expected,
                pos: new_pos,
                outside: outside.rotate_rev(),
            };
        }
    }

    // step sideways or rotate in place
    let step_dir = outside.rotate();
    let neighbor = plot.move_pos(pos, step_dir);
    match neighbor {
        Some(new_pos @ Pos(x, y)) if plot.data[y][x].plant == expected => BorderState {
            expected,
            pos: new_pos,
            outside,
        },
        _ => BorderState {
            expected,
            pos,
            outside: outside.rotate(),
        },
    }
}

fn mark_side(pos: Pos, outside: Direction, plot: &mut Plot) -> usize {
    let start = plot.get(pos).expect("start within plot");
    let expected = start.plant;

    if start.is_marked(outside) {
        return 0;
    }

    let not_actually_outside = plot
        .move_pos(pos, outside)
        .is_some_and(|p| plot.get(p).expect("checked").plant == expected);
    if not_actually_outside {
        return 0;
    }

    let first_state = BorderState {
        expected,
        pos,
        outside,
    };

    let mut sides = 0;
    let mut current_state = first_state;

    let mut init = true;
    let mut step = 0;
    while current_state != first_state || init {
        init = false;
        step += 1;
        assert!(step < 10000);

        let next_state = get_next_state(current_state, plot);
        if next_state.outside != current_state.outside {
            sides += 1;
        }
        plot.get_mut(next_state.pos)
            .expect("checked")
            .mark(next_state.outside);

        current_state = next_state;
    }

    sides
}

fn mark_all_sides(pos: Pos, plot: &mut Plot) -> usize {
    let start = plot.get(pos).expect("start within plot");

    let Pos(x, y) = pos;
    let expected = start.plant;

    let outside = None
        .or_else(|| (plot.move_pos(pos, Direction::Left).is_none()).then_some(Direction::Left))
        .or_else(|| (plot.move_pos(pos, Direction::Right).is_none()).then_some(Direction::Right))
        .or_else(|| (plot.move_pos(pos, Direction::Up).is_none()).then_some(Direction::Up))
        .or_else(|| (plot.move_pos(pos, Direction::Down).is_none()).then_some(Direction::Down));
    let outside = match outside {
        Some(dir) => dir,
        None => None
            .or_else(|| (plot.data[y][x - 1].plant != expected).then_some(Direction::Left))
            .or_else(|| (plot.data[y][x + 1].plant != expected).then_some(Direction::Right))
            .or_else(|| (plot.data[y + 1][x].plant != expected).then_some(Direction::Up))
            .or_else(|| (plot.data[y - 1][x].plant != expected).then_some(Direction::Down))
            .expect("we are at border"),
    };
    if start.is_marked(outside) {
        return 0;
    }

    const DIRS: [Direction; 4] = [
        Direction::Up,
        Direction::Right,
        Direction::Down,
        Direction::Left,
    ];

    let mut total = 0;
    for &outside in DIRS.iter() {
        total += mark_side(pos, outside, plot);
    }

    total
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Pos(usize, usize);

#[derive(Debug)]
struct BFSState {
    last: Option<Pos>,
    next: Option<Pos>,
}

fn get_bulk_price_at(pos: Pos, plot: &mut Plot) -> usize {
    let Pos(x, y) = pos;
    assert!(x < plot.width && y < plot.height);
    let expected = match plot.data[y][x] {
        Tile { visited, .. } if visited => return 0,
        Tile { plant, .. } => plant,
    };

    let mut sides = 0;
    let mut fields = 0;

    let mut queue = VecDeque::new();
    queue.push_back(BFSState {
        last: None,
        next: Some(pos),
    });
    while !queue.is_empty() {
        let BFSState { last, next } = queue.pop_front().expect("checked not empty");
        let tile = next.map(|p| plot.get_mut(p).expect("only valid pos in queue"));
        match tile {
            None => {
                let last = last.expect("no longer start");
                let new_sides = mark_all_sides(last, plot);
                sides += new_sides;
                continue;
            }
            Some(Tile { plant, .. }) if *plant != expected => {
                let last = last.expect("no longer start");
                let new_sides = mark_all_sides(last, plot);
                sides += new_sides;
                continue;
            }
            Some(Tile { visited, .. }) if !*visited => {
                *visited = true;
                fields += 1;
            }
            Some(_) => {
                continue;
            }
        }

        let last = next;
        let Pos(x, y) = next.expect("checked that it is valid");
        let left = BFSState {
            last,
            next: (x > 0).then(|| Pos(x - 1, y)),
        };
        let up = BFSState {
            last,
            next: (y > 0).then(|| Pos(x, y - 1)),
        };
        let right = BFSState {
            last,
            next: (x < plot.width - 1).then(|| Pos(x + 1, y)),
        };
        let down = BFSState {
            last,
            next: (y < plot.height - 1).then(|| Pos(x, y + 1)),
        };
        queue.push_back(left);
        queue.push_back(up);
        queue.push_back(right);
        queue.push_back(down);
    }

    let price = sides * fields;
    price
}

fn get_perimeter_price_bulk(mut plot: Plot) -> usize {
    let mut price = 0;
    for y in 0..plot.height {
        for x in 0..plot.width {
            price += get_bulk_price_at(Pos(x, y), &mut plot);
        }
    }
    price
}

pub fn run() -> Result<()> {
    println!("day 12");
    let path = PathBuf::from("./resources/day12.txt");
    let data = util::get_data_string(&path)?;
    let plot = parse_plot(&data)?;
    let price = get_perimeter_price(plot.clone());
    println!("price for perimeter: {price}");
    let bulk_price = get_perimeter_price_bulk(plot);
    println!("price for perimeter in bulk: {bulk_price}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_plot() {
        let input = "AAAA
BBCD
BBCC
EEEC";
        let plot = parse_plot(input).unwrap();
        let price = get_perimeter_price(plot);
        assert_eq!(price, 140);
    }

    #[test]
    fn test_plot_with_holes() {
        let input = "OOOOO
OXOXO
OOOOO
OXOXO
OOOOO";
        let plot = parse_plot(input).unwrap();
        let price = get_perimeter_price(plot);
        assert_eq!(price, 772);
    }

    #[test]
    fn test_simple_plot_bulk() {
        let input = "AAAA
BBCD
BBCC
EEEC";
        let plot = parse_plot(input).unwrap();
        let price = get_perimeter_price_bulk(plot);
        assert_eq!(price, 80);
    }

    #[test]
    fn test_plot_with_holes_bulk() {
        let input = "OOOOO
OXOXO
OOOOO
OXOXO
OOOOO";
        let plot = parse_plot(input).unwrap();
        let price = get_perimeter_price_bulk(plot);
        assert_eq!(price, 436);
    }
}

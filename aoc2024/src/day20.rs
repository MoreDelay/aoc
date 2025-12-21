use anyhow::Result;
use std::{collections::VecDeque, ops::RangeInclusive, path::PathBuf};

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Pos(usize, usize);

#[derive(Copy, Clone, Debug)]
enum Tile {
    Empty(usize),
    Wall,
    Start,
    End(usize),
}

#[allow(dead_code)]
struct RaceTrack {
    size: Pos,
    start: Pos,
    end: Pos,
    tiles: Vec<Tile>,
}

impl RaceTrack {
    fn parse(input: &str) -> Result<RaceTrack> {
        use AocError::ParseError;
        use Tile::*;
        let cap = input.chars().filter(|&c| c != '\n').count();
        let mut tiles = Vec::with_capacity(cap);
        let mut start = None;
        let mut end = None;
        let mut width = None;
        let mut height = 0;

        for (y, l) in input.split("\n").filter(|s| !s.is_empty()).enumerate() {
            height += 1;
            if let Some(w) = width {
                if w != l.len() {
                    return Err(ParseError.into());
                }
            } else {
                width = Some(l.len());
            }

            for (x, c) in l.chars().enumerate() {
                let tile = match c {
                    '.' => Empty(0),
                    '#' => Wall,
                    'S' if start.is_none() => {
                        start = Some(Pos(x, y));
                        Start
                    }
                    'E' if end.is_none() => {
                        end = Some(Pos(x, y));
                        End(0)
                    }
                    _ => return Err(ParseError.into()),
                };
                tiles.push(tile);
            }
        }

        let width = width.ok_or(ParseError)?;
        let start = start.ok_or(ParseError)?;
        let end = end.ok_or(ParseError)?;

        let size = Pos(width, height);
        let mut race_track = RaceTrack {
            size,
            start,
            end,
            tiles,
        };
        race_track.calculate_distances();
        Ok(race_track)
    }

    fn at(&self, pos: Pos) -> Tile {
        let Pos(width, _) = self.size;
        let Pos(x, y) = pos;
        self.tiles[y * width + x]
    }

    fn at_mut(&mut self, pos: Pos) -> &mut Tile {
        let Pos(width, _) = self.size;
        let Pos(x, y) = pos;
        &mut self.tiles[y * width + x]
    }

    fn calculate_distances(&mut self) {
        struct BFSNode {
            pos: Pos,
            dist: usize,
        }
        let initial = BFSNode {
            pos: self.start,
            dist: 0,
        };
        let mut queue = VecDeque::new();
        queue.push_back(initial);

        let Pos(width, height) = self.size;

        while let Some(node) = queue.pop_front() {
            assert!(queue.len() < width * height);
            let BFSNode { pos, dist } = node;
            let tile = self.at(pos);
            let prev_dist = match tile {
                Tile::Empty(distance) => distance,
                Tile::Wall => continue,
                Tile::Start => 0,
                Tile::End(distance) => distance,
            };
            if prev_dist > 0 && !matches!(tile, Tile::Start) {
                continue;
            }

            match tile {
                Tile::Empty(_) => *self.at_mut(pos) = Tile::Empty(dist),
                Tile::End(_) => *self.at_mut(pos) = Tile::End(dist),
                Tile::Start => (),
                _ => unreachable!(),
            };
            let Pos(x, y) = pos;
            let dist = dist + 1;
            if x > 0 {
                let pos = Pos(x - 1, y);
                queue.push_back(BFSNode { pos, dist });
            }
            if y > 0 {
                let pos = Pos(x, y - 1);
                queue.push_back(BFSNode { pos, dist });
            }
            if x < width - 1 {
                let pos = Pos(x + 1, y);
                queue.push_back(BFSNode { pos, dist });
            }
            if y < height - 1 {
                let pos = Pos(x, y + 1);
                queue.push_back(BFSNode { pos, dist });
            }
        }
    }

    fn iter_pos(&self) -> TrackPosIter<'_> {
        TrackPosIter::new(self)
    }

    fn iter_circle(&self, origin: Pos, radius: usize) -> CircleIter {
        CircleIter::new(self.size, origin, radius)
    }

    fn iter_cheats(&self, length: usize) -> CheatsIter<'_> {
        CheatsIter::new(self, length)
    }
}

struct TrackPosIter<'a> {
    track: &'a RaceTrack,
    pos: Option<Pos>,
    dist: usize,
}

impl<'a> TrackPosIter<'a> {
    fn new(track: &'a RaceTrack) -> Self {
        let pos = Some(track.start);
        let dist = 0;
        TrackPosIter { track, pos, dist }
    }
}

impl Iterator for TrackPosIter<'_> {
    type Item = Pos;

    fn next(&mut self) -> Option<Self::Item> {
        let out = self.pos.take();
        let Pos(x, y) = out?;
        let Pos(width, height) = self.track.size;

        let is_next = |p| {
            let dist = match self.track.at(p) {
                Tile::Empty(dist) => dist,
                Tile::Wall => return false,
                Tile::Start => 0,
                Tile::End(dist) => dist,
            };
            dist > self.dist
        };

        let next_pos = match () {
            () if x > 0 && is_next(Pos(x - 1, y)) => Pos(x - 1, y),
            () if y > 0 && is_next(Pos(x, y - 1)) => Pos(x, y - 1),
            () if x < width - 1 && is_next(Pos(x + 1, y)) => Pos(x + 1, y),
            () if y < height - 1 && is_next(Pos(x, y + 1)) => Pos(x, y + 1),
            _ => {
                let end = self.track.at(Pos(x, y));
                assert!(matches!(end, Tile::End(_)));
                return out;
            }
        };
        self.pos = Some(next_pos);
        self.dist += 1;
        out
    }
}

struct CircleIter {
    size: Pos,
    origin: Pos,
    radius: isize,
    delta_y: isize,
    did_left: bool,
}

impl CircleIter {
    fn new(size: Pos, origin: Pos, radius: usize) -> CircleIter {
        let radius = radius as isize;
        let delta_y = -radius;
        let did_left = false;
        CircleIter {
            size,
            origin,
            radius,
            delta_y,
            did_left,
        }
    }
}

impl Iterator for CircleIter {
    type Item = Pos;

    fn next(&mut self) -> Option<Self::Item> {
        let &mut CircleIter {
            size: Pos(width, height),
            origin: Pos(x_o, y_o),
            radius,
            ..
        } = self;
        while self.delta_y <= radius {
            let delta_y = self.delta_y;
            let pos_per_side = radius - self.delta_y.abs();
            // set state for next iteration
            let delta_x = if pos_per_side > 0 && !self.did_left {
                self.did_left = true;
                -pos_per_side
            } else {
                self.delta_y += 1;
                self.did_left = false;
                pos_per_side
            };
            let out_of_bounds = delta_y < 0 && (-delta_y) as usize > y_o
                || delta_x < 0 && (-delta_x) as usize > x_o
                || delta_y >= 0 && y_o + delta_y as usize >= height
                || delta_x >= 0 && x_o + delta_x as usize >= width;
            if out_of_bounds {
                continue;
            }
            let out_x = match delta_x >= 0 {
                true => x_o + delta_x as usize,
                false => x_o - (-delta_x) as usize,
            };
            let out_y = match delta_y >= 0 {
                true => y_o + delta_y as usize,
                false => y_o - (-delta_y) as usize,
            };

            return Some(Pos(out_x, out_y));
        }
        None
    }
}

#[allow(dead_code)]
struct Cheat {
    start: Pos,
    end: Pos,
    saving: usize,
}

struct CheatsIter<'a> {
    track: &'a RaceTrack,
    length: usize,
    last_pos: Option<Pos>,
    last_radius: Option<usize>,
    track_iter: TrackPosIter<'a>,
    radius_iter: RangeInclusive<usize>,
    circle_iter: CircleIter,
}

impl<'a> CheatsIter<'a> {
    fn new(track: &'a RaceTrack, length: usize) -> Self {
        assert!(length > 0);
        let mut track_iter = track.iter_pos();
        let last_pos = track_iter.next();

        let mut radius_iter = 1..=length;
        let last_radius = radius_iter.next();

        let circle_iter = track.iter_circle(
            last_pos.expect("track has start"),
            last_radius.expect("length >= 1"),
        );

        CheatsIter {
            track,
            length,
            last_pos,
            last_radius,
            track_iter,
            radius_iter,
            circle_iter,
        }
    }

    fn make_cheat(&self, end: Pos) -> Option<Cheat> {
        let start = self.last_pos.expect("called during valid iteration");
        let radius = self.last_radius.expect("called during valid iteration");

        let at_start = match self.track.at(start) {
            Tile::Empty(d) => d,
            Tile::Start => 0,
            Tile::End(d) => d,
            Tile::Wall => unreachable!(),
        };

        let at_end = match self.track.at(end) {
            Tile::Empty(d) => d,
            Tile::Start => return None,
            Tile::End(d) => d,
            Tile::Wall => return None,
        };
        if at_start + radius >= at_end {
            return None;
        }
        let saving = at_end - at_start - radius;
        Some(Cheat { start, end, saving })
    }

    fn update_radius(&mut self) {
        let Some(start) = self.last_pos else {
            self.last_radius = None;
            return;
        };
        self.last_radius = self.radius_iter.next();
        if let Some(radius) = self.last_radius {
            self.circle_iter = self.track.iter_circle(start, radius);
        }
    }

    fn update_pos(&mut self) {
        self.last_pos = self.track_iter.next();
        if self.last_pos.is_some() {
            self.radius_iter = 1..=self.length;
        }
    }
}

impl Iterator for CheatsIter<'_> {
    type Item = Cheat;

    fn next(&mut self) -> Option<Self::Item> {
        while self.last_pos.is_some() {
            while self.last_radius.is_some() {
                while let Some(end) = self.circle_iter.next() {
                    if let Some(cheat) = self.make_cheat(end) {
                        return Some(cheat);
                    }
                }
                self.update_radius();
            }
            self.update_pos();
            self.update_radius();
        }
        None
    }
}

fn count_good_cheats<T>(cheats: T, lower_bound: usize) -> usize
where
    T: Iterator<Item = Cheat>,
{
    cheats
        .filter(|&Cheat { saving, .. }| saving >= lower_bound)
        .count()
}

pub fn run() -> Result<()> {
    println!("day 20");
    let path = PathBuf::from("./resources/day20.txt");
    let data = util::get_data_string(&path)?;
    let track = RaceTrack::parse(&data)?;
    let cheat_iter = track.iter_cheats(2);
    let good_cheats = count_good_cheats(cheat_iter, 100);
    println!("good 2 picosecond cheating spots: {good_cheats}");
    let cheat_iter = track.iter_cheats(20);
    let good_cheats = count_good_cheats(cheat_iter, 100);
    println!("good 20 picosecond cheating spots: {good_cheats}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_cheats() {
        let input = "###############
#...#...#.....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#######.#.#.###
#######.#.#...#
#######.#.###.#
###..E#...#...#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############";
        let track = RaceTrack::parse(input).unwrap();

        let with_saving = |expected| {
            track
                .iter_cheats(2)
                .filter(|&Cheat { saving, .. }| saving == expected)
                .count()
        };

        assert_eq!(track.iter_cheats(2).count(), 44);
        assert_eq!(with_saving(2), 14);
        assert_eq!(with_saving(4), 14);
        assert_eq!(with_saving(6), 2);
        assert_eq!(with_saving(8), 4);
        assert_eq!(with_saving(10), 2);
        assert_eq!(with_saving(12), 3);
        assert_eq!(with_saving(20), 1);
        assert_eq!(with_saving(36), 1);
        assert_eq!(with_saving(38), 1);
        assert_eq!(with_saving(40), 1);
        assert_eq!(with_saving(64), 1);
    }

    #[test]
    fn test_large_cheats() {
        let input = "###############
#...#...#.....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#######.#.#.###
#######.#.#...#
#######.#.###.#
###..E#...#...#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############";
        let track = RaceTrack::parse(input).unwrap();
        let with_saving = |expected| {
            track
                .iter_cheats(20)
                .filter(|&Cheat { saving, .. }| saving == expected)
                .count()
        };

        assert_eq!(with_saving(50), 32);
        assert_eq!(with_saving(52), 31);
        assert_eq!(with_saving(54), 29);
        assert_eq!(with_saving(56), 39);
        assert_eq!(with_saving(58), 25);
        assert_eq!(with_saving(60), 23);
        assert_eq!(with_saving(62), 20);
        assert_eq!(with_saving(64), 19);
        assert_eq!(with_saving(66), 12);
        assert_eq!(with_saving(68), 14);
        assert_eq!(with_saving(70), 12);
        assert_eq!(with_saving(72), 22);
        assert_eq!(with_saving(74), 4);
        assert_eq!(with_saving(76), 3);
    }

    #[test]
    fn test_circle_iter() {
        let size = Pos(21, 21);
        let origin = Pos(10, 10);
        let radius = 3;
        let mut iter = CircleIter::new(size, origin, radius);
        assert_eq!(iter.next(), Some(Pos(10, 7)));
        assert_eq!(iter.next(), Some(Pos(9, 8)));
        assert_eq!(iter.next(), Some(Pos(11, 8)));
        assert_eq!(iter.next(), Some(Pos(8, 9)));
        assert_eq!(iter.next(), Some(Pos(12, 9)));
        assert_eq!(iter.next(), Some(Pos(7, 10)));
        assert_eq!(iter.next(), Some(Pos(13, 10)));
        assert_eq!(iter.next(), Some(Pos(8, 11)));
        assert_eq!(iter.next(), Some(Pos(12, 11)));
        assert_eq!(iter.next(), Some(Pos(9, 12)));
        assert_eq!(iter.next(), Some(Pos(11, 12)));
        assert_eq!(iter.next(), Some(Pos(10, 13)));
        assert_eq!(iter.next(), None);
    }
}

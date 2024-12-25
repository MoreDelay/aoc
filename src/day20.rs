use anyhow::Result;
use std::{collections::VecDeque, path::PathBuf};

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

    fn iter(&self) -> TrackIter {
        TrackIter {
            track: self,
            pos: Some(self.start),
            dist: 0,
        }
    }
}

struct TrackIter<'a> {
    track: &'a RaceTrack,
    pos: Option<Pos>,
    dist: usize,
}

impl Iterator for TrackIter<'_> {
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
            _ => return out,
        };
        self.pos = Some(next_pos);
        self.dist += 1;
        out
    }
}

#[allow(dead_code)]
struct Cheat {
    start: Pos,
    end: Pos,
    saving: usize,
}

fn find_all_cheats(track: &RaceTrack) -> Vec<Cheat> {
    let Pos(width, height) = track.size;
    let mut cheats = Vec::new();

    for start in track.iter() {
        let Pos(x, y) = start;
        let at_start = match track.at(start) {
            Tile::Empty(d) => d,
            Tile::Start => 0,
            Tile::End(d) => d,
            Tile::Wall => unreachable!(),
        };
        let make_cheat = |end| {
            let at_end = match track.at(end) {
                Tile::Empty(d) => d,
                Tile::Start => return None,
                Tile::End(d) => d,
                Tile::Wall => return None,
            };
            if at_start >= at_end {
                return None;
            }
            let saving = at_end - at_start - 2;
            if saving == 0 {
                return None;
            }
            Some(Cheat { start, end, saving })
        };
        if x > 1 {
            let end = Pos(x - 2, y);
            if let Some(cheat) = make_cheat(end) {
                cheats.push(cheat);
            }
        }
        if y > 1 {
            let end = Pos(x, y - 2);
            if let Some(cheat) = make_cheat(end) {
                cheats.push(cheat);
            }
        }
        if x < width - 2 {
            let end = Pos(x + 2, y);
            if let Some(cheat) = make_cheat(end) {
                cheats.push(cheat);
            }
        }
        if y < height - 2 {
            let end = Pos(x, y + 2);
            if let Some(cheat) = make_cheat(end) {
                cheats.push(cheat);
            }
        }
    }

    cheats
}

fn count_good_cheats(cheats: &[Cheat], lower_bound: usize) -> usize {
    cheats
        .iter()
        .filter(|&&Cheat { saving, .. }| saving >= lower_bound)
        .count()
}

pub fn run() -> Result<()> {
    println!("day 20");
    let path = PathBuf::from("./resources/day20.txt");
    let data = util::get_data_string(&path)?;
    let track = RaceTrack::parse(&data)?;
    let cheats = find_all_cheats(&track);
    let good_cheats = count_good_cheats(&cheats, 100);
    println!("good cheating spots: {good_cheats}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_patterns() {
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
        let cheats = find_all_cheats(&track);

        let with_saving = |expected| {
            cheats
                .iter()
                .filter(|&&Cheat { saving, .. }| saving == expected)
                .count()
        };

        assert_eq!(cheats.len(), 44);
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
}

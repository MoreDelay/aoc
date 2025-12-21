use std::{collections::BTreeSet, num::ParseIntError};

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct FreshRange {
    start: usize,
    end: usize,
}

#[derive(Debug, Error)]
enum FreshRangeParseError {
    #[error("expected format '<number>-<number>', got {0}")]
    WrongFormat(String),
    #[error("left number must be smaller than right, got {0}-{1}")]
    IncorrectRange(usize, usize),
}

impl FreshRange {
    fn parse(input: &str) -> Result<Self, FreshRangeParseError> {
        let input = input.trim();
        let split_index = input
            .find('-')
            .ok_or_else(|| FreshRangeParseError::WrongFormat(input.to_string()))?;
        let (start, end) = input.split_at(split_index);
        let end = &end[1..]; // remove '-'

        let start = start
            .parse::<usize>()
            .map_err(|_| FreshRangeParseError::WrongFormat(input.to_string()))?;
        let end = end
            .parse::<usize>()
            .map_err(|_| FreshRangeParseError::WrongFormat(input.to_string()))?;

        let good = start <= end;
        if !good {
            return Err(FreshRangeParseError::IncorrectRange(start, end));
        }

        Ok(Self { start, end })
    }

    fn contains(&self, v: &usize) -> bool {
        (self.start..=self.end).contains(v)
    }
}

struct Database {
    ranges: Vec<FreshRange>,
    ingredients: Vec<usize>,
}

#[derive(Debug, Error)]
enum DatabaseParseError {
    #[error("error while reading")]
    ReadError(#[from] std::io::Error),
    #[error("wrong format on line {0}")]
    WrongRangeFormat(usize, #[source] FreshRangeParseError),
    #[error("wrong format on line {0}")]
    WrongIngredientFormat(usize, #[source] ParseIntError),
    #[error("an unexpected section begins at line {0}")]
    UnexpectedSection(usize),
}

impl Database {
    fn parse(input: impl std::io::BufRead) -> Result<Self, DatabaseParseError> {
        let mut lines_iter = input.lines().enumerate();

        let mut ranges = Vec::new();
        for (i, line) in lines_iter.by_ref() {
            let line = line?;
            if line.is_empty() {
                // done with ranges
                break;
            }

            let range =
                FreshRange::parse(&line).map_err(|e| DatabaseParseError::WrongRangeFormat(i, e))?;
            ranges.push(range);
        }

        let mut ingredients = Vec::new();
        for (i, line) in lines_iter.by_ref() {
            let line = line?;
            if line.is_empty() {
                // done with ingredients
                break;
            }

            let ingredient = line
                .parse()
                .map_err(|e| DatabaseParseError::WrongIngredientFormat(i, e))?;
            ingredients.push(ingredient);
        }

        for (i, line) in lines_iter.by_ref() {
            let line = line?;
            if !line.is_empty() {
                // expect no other data at this point
                return Err(DatabaseParseError::UnexpectedSection(i));
            }
        }

        Ok(Self {
            ranges,
            ingredients,
        })
    }

    fn count_fresh_ingredients(&self) -> usize {
        self.ingredients
            .iter()
            .filter(|id| self.ranges.iter().any(|range| range.contains(id)))
            .count()
    }
}

struct MinimizedRanges(BTreeSet<FreshRange>);

impl MinimizedRanges {
    fn from_database(database: &Database) -> Self {
        let mut set = BTreeSet::<FreshRange>::new();

        for outside in database.ranges.iter() {
            let extracted = set.extract_if(.., |within| {
                let contained_within =
                    within.contains(&outside.start) || within.contains(&outside.end);
                let contained_outside =
                    outside.contains(&within.start) || outside.contains(&within.end);
                contained_within || contained_outside
            });

            let simplified = extracted.fold(*outside, |acc, extracted| FreshRange {
                start: acc.start.min(extracted.start),
                end: acc.end.max(extracted.end),
            });

            set.insert(simplified);
        }

        Self(set)
    }
}

struct SimplifiedDatabase(Database);

impl SimplifiedDatabase {
    fn simplify(database: Database) -> Self {
        let minimized_ranges = MinimizedRanges::from_database(&database);
        let ranges = minimized_ranges.0.into_iter().collect();
        let database = Database { ranges, ..database };
        Self(database)
    }

    fn count_fresh_ids(&self) -> usize {
        self.0.ranges.iter().map(|r| r.end - r.start + 1).sum()
    }
}

pub fn day05() -> anyhow::Result<()> {
    let path = std::path::PathBuf::from("resources/day05.txt");
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let database = Database::parse(reader)?;
    let count = database.count_fresh_ingredients();
    println!("day05: number of fresh ingredients is {count}");
    let database = SimplifiedDatabase::simplify(database);
    let count = database.count_fresh_ids();
    println!("day05: number of fresh ingredient ids is {count}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    static EXAMPLE: &str = "3-5
10-14
16-20
12-18

1
5
8
11
17
32";

    #[test]
    fn test_input_01() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let database = Database::parse(cursor)?;
        let count = database.count_fresh_ingredients();
        assert_eq!(count, 3);
        Ok(())
    }

    #[test]
    fn test_input_02() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let database = Database::parse(cursor)?;
        let database = SimplifiedDatabase::simplify(database);
        let count = database.count_fresh_ids();
        assert_eq!(count, 14);
        Ok(())
    }
}

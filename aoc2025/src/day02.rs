use thiserror::Error;

#[derive(Debug, Copy, Clone, PartialEq)]
struct Id {
    value: usize,
}

#[derive(Error, Debug)]
pub enum IdParseError {
    #[error("expected format '<number>', got {0}")]
    WrongFormat(String),
}

impl Id {
    fn parse(input: &str) -> Result<Self, IdParseError> {
        Ok(Self {
            value: input
                .parse()
                .map_err(|_| IdParseError::WrongFormat(input.to_string()))?,
        })
    }

    fn is_valid(&self, max_split: Option<usize>) -> bool {
        let string = self.value.to_string();
        let len = string.len();

        let max_split = max_split.map(|v| v.min(len)).unwrap_or(len);

        'outer: for splits in 2..=max_split {
            if !len.is_multiple_of(splits) {
                continue;
            }
            let split_len = len / splits;
            let (check, mut rest) = string.split_at(split_len);
            for _ in 2..=splits {
                let (cur, right) = rest.split_at(split_len);
                assert!(cur.len() == split_len);

                if cur != check {
                    continue 'outer;
                }
                rest = right;
            }
            assert!(rest.is_empty());

            return false;
        }

        true
    }
}

#[derive(Debug, Copy, Clone)]
struct IdRange {
    start: Id,
    end: Id,
}

#[derive(thiserror::Error, Debug)]
pub enum IdRangeParseError {
    #[error("expected format '<number>-<number>', got {0}")]
    MissingIdPair(String),
    #[error("one id is wrong")]
    IdFormatWrong(#[from] IdParseError),
}

impl IdRange {
    fn parse(input: &str) -> Result<Self, IdRangeParseError> {
        let split_index = input
            .find('-')
            .ok_or(IdRangeParseError::MissingIdPair(input.to_string()))?;
        let (start, end) = input.split_at(split_index);
        let end = &end[1..];
        let start = Id::parse(start)?;
        let end = Id::parse(end)?;
        Ok(Self { start, end })
    }

    fn iter(&self) -> IdRangeIter {
        IdRangeIter {
            next: Some(self.start),
            last: self.end,
        }
    }
}

struct IdRangeIter {
    next: Option<Id>,
    last: Id,
}

impl Iterator for IdRangeIter {
    type Item = Id;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next?;
        if next == self.last {
            self.next = None;
        } else {
            self.next = Some(Id {
                value: next.value + 1,
            });
        }
        Some(next)
    }
}

#[derive(Debug)]
struct IdRanges(Vec<IdRange>);

#[derive(thiserror::Error, Debug)]
pub enum IdRangesParseError {
    #[error("error while reading")]
    ReadError(#[from] std::io::Error),
    #[error("not uft8")]
    Utf8Error(#[from] std::string::FromUtf8Error),
    #[error("input has wrong format at line {0}")]
    WrongFormat(usize, #[source] IdRangeParseError),
}

impl IdRanges {
    fn parse(mut input: impl std::io::BufRead) -> Result<Self, IdRangesParseError> {
        let mut ranges = Vec::new();

        let mut buf = Vec::new();
        while input.read_until(b',', &mut buf)? != 0 {
            let string = String::from_utf8(buf)?;
            let slice = if string.ends_with(',') || string.ends_with('\n') {
                &string[..string.len() - 1]
            } else {
                &string
            };
            let range = IdRange::parse(slice).map_err(|e| IdRangesParseError::WrongFormat(0, e))?;
            ranges.push(range);

            buf = string.into_bytes();
            buf.clear();
        }
        Ok(IdRanges(ranges))
    }

    fn sum_of_invaid_ids_two_split(&self) -> usize {
        self.0
            .iter()
            .flat_map(|range| range.iter())
            .filter(|id| !id.is_valid(Some(2)))
            .map(|id| id.value)
            .sum()
    }

    fn sum_of_invaid_ids_n_split(&self) -> usize {
        self.0
            .iter()
            .flat_map(|range| range.iter())
            .filter(|id| !id.is_valid(None))
            .map(|id| id.value)
            .sum()
    }
}

#[derive(Debug, Error)]
pub enum Day02Error {
    #[error("could not open file")]
    FileError(#[from] std::io::Error),
    #[error("could not parse input")]
    ParseError(#[from] IdRangesParseError),
}

pub fn day02() -> Result<(), Day02Error> {
    let path = std::path::PathBuf::from("resources/day02.txt");
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let input = IdRanges::parse(reader)?;
    let sum_two_split = input.sum_of_invaid_ids_two_split();
    println!("day02: sum of invalid ids symmetric in 2 splits is {sum_two_split}");
    let sum_n_split = input.sum_of_invaid_ids_n_split();
    println!("day02: sum of invalid ids symmetric in n splits is {sum_n_split}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    static EXAMPLE: &str = "11-22,95-115,998-1012,1188511880-1188511890,222220-222224,1698522-1698528,446443-446449,38593856-38593862,565653-565659,824824821-824824827,2121212118-2121212124";

    #[test]
    fn test_input_01() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let input = IdRanges::parse(cursor)?;
        let sum = input.sum_of_invaid_ids_two_split();
        assert_eq!(sum, 1227775554);
        Ok(())
    }

    #[test]
    fn test_input_02() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let input = IdRanges::parse(cursor)?;
        let sum = input.sum_of_invaid_ids_n_split();
        assert_eq!(sum, 4174379265);
        Ok(())
    }
}

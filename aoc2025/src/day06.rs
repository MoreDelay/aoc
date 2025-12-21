use std::{io::Read, num::ParseIntError};

use thiserror::Error;

enum Task {
    Addition(Vec<usize>),
    Multiplication(Vec<usize>),
}

struct Worksheet {
    tasks: Vec<Task>,
}

#[derive(Debug, Error)]
pub enum WorksheetParseError {
    #[error("expected even grid of numbers and operators, missing value at row {0}, col {1}")]
    UnevenGrid(usize, usize),
    #[error("expect at least one number and one operator per column, but only got {0} rows")]
    IncompleteTask(usize),
    #[error("expect grid of numbers, got '{2}' at row {0}, col {1}")]
    UnexpectedGridNumber(usize, usize, String, #[source] ParseIntError),
    #[error("expect grid of numbers, got '{2}' at row {0}, col {1}")]
    UnexpectedGridNumberChar(usize, usize, char),
    #[error("expect grid of numbers, got '{2}' at number column {0}, grid col {1}")]
    UnexpectedGridNumberColumn(usize, usize, String, #[source] ParseIntError),
    #[error("row of operators '*' or '+', got '{2}' at row {0}, col {1}")]
    UnexpectedOperator(usize, usize, String),
    #[error("found two operators in the same column, first '{2}', then '{3}', at row {0}, col {1}")]
    OperatorTwiceInColumn(usize, usize, char, char),
    #[error("no operator found in row {0}, col {1}")]
    OperatorMissingInColumn(usize, usize),
}

impl Worksheet {
    fn parse_numbers_rowwise(input: &str) -> Result<Self, WorksheetParseError> {
        let mut line_iters = input
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| line.split_ascii_whitespace())
            .collect::<Vec<_>>();

        if line_iters.len() < 2 {
            return Err(WorksheetParseError::IncompleteTask(line_iters.len()));
        }

        let mut tasks = Vec::new();

        let mut col = 0;
        loop {
            let values = line_iters
                .iter_mut()
                .map(|iter| iter.next())
                .collect::<Vec<_>>();

            let all_missing = values.iter().all(|v| v.is_none());
            if all_missing {
                // all iterators exhausted at once
                break;
            }

            let mut values = values
                .into_iter()
                .enumerate()
                .map(|(row, v)| v.ok_or(WorksheetParseError::UnevenGrid(row, col)))
                .collect::<Result<Vec<_>, _>>()?;

            let op = values
                .pop()
                .expect("checked that values has length at least 2");
            let values = values
                .into_iter()
                .enumerate()
                .map(|(row, v)| {
                    v.parse::<usize>().map_err(|e| {
                        WorksheetParseError::UnexpectedGridNumber(row, col, v.to_string(), e)
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;

            let task = match op {
                "*" => Task::Multiplication(values),
                "+" => Task::Addition(values),
                _ => {
                    let row = values.len(); // operator row comes after all values 
                    let op = op.to_string();
                    return Err(WorksheetParseError::UnexpectedOperator(row, col, op));
                }
            };

            tasks.push(task);
            col += 1;
        }

        Ok(Self { tasks })
    }

    fn parse_numbers_colwise(input: &str) -> Result<Self, WorksheetParseError> {
        let mut line_iters = input
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| line.chars())
            .collect::<Vec<_>>();

        if line_iters.len() < 2 {
            return Err(WorksheetParseError::IncompleteTask(line_iters.len()));
        }
        let op_row = line_iters.len() - 1;

        #[derive(Default)]
        struct ParsingState {
            tasks: Vec<Task>,
            // the data for the task we are building right now
            numbers: Vec<usize>,
            op: Option<char>,
            // information on location in input for error reporting
            grid_col: usize,
            num_col: usize,
        }

        let mut state = ParsingState::default();

        loop {
            // create a const copy for this iteration so that the finalizing closure can modify the
            // mutable value for the next iteration (without using RefCell)
            let grid_col = state.grid_col;

            let mut finalize_task = || {
                // create a clone and clear original vector to keep reserved allocation
                let numbers = state.numbers.clone();
                state.numbers.clear();
                let op = state.op.take();
                state.grid_col += 1;
                state.num_col = 0;

                assert!(
                    !numbers.is_empty(),
                    "never happens as at least one number column is above the operator"
                );
                let task = match op {
                    Some('*') => Task::Multiplication(numbers),
                    Some('+') => Task::Addition(numbers),
                    None => {
                        return Err(WorksheetParseError::OperatorMissingInColumn(
                            op_row, grid_col,
                        ));
                    }
                    _ => unreachable!(),
                };
                state.tasks.push(task);
                Ok(())
            };

            let chars = line_iters
                .iter_mut()
                .map(|iter| iter.next())
                .collect::<Vec<_>>();

            let all_missing = chars.iter().all(|v| v.is_none());
            if all_missing {
                // all iterators exhausted at once
                finalize_task()?;
                return Ok(Self { tasks: state.tasks });
            }

            let mut chars = chars
                .into_iter()
                .enumerate()
                .map(|(row, v)| v.ok_or(WorksheetParseError::UnevenGrid(row, grid_col)))
                .collect::<Result<Vec<_>, _>>()?;

            let all_space = chars.iter().all(|&v| v == ' ');
            if all_space {
                // complete task and move to next column
                finalize_task()?;
                continue;
            }

            let op = chars
                .pop()
                .expect("checked that values has length at least 2");
            let digits = chars;

            match op {
                ' ' => (),
                '+' | '*' => {
                    if let Some(state_op) = state.op {
                        return Err(WorksheetParseError::OperatorTwiceInColumn(
                            op_row, grid_col, state_op, op,
                        ));
                    }
                    state.op = Some(op)
                }
                _ => {
                    let op = op.to_string();
                    return Err(WorksheetParseError::UnexpectedOperator(
                        op_row, grid_col, op,
                    ));
                }
            }

            let number = digits
                .into_iter()
                .enumerate()
                .filter(|(_, c)| *c != ' ')
                .map(|(row, c)| match c.is_ascii_digit() {
                    true => Ok(c),
                    false => Err(WorksheetParseError::UnexpectedGridNumberChar(
                        row, grid_col, c,
                    )),
                })
                .collect::<Result<String, _>>()?;

            let number = number.parse::<usize>().map_err(|e| {
                WorksheetParseError::UnexpectedGridNumberColumn(state.num_col, grid_col, number, e)
            })?;

            state.numbers.push(number);
            state.num_col += 1;
        }
    }

    fn compute_grand_total(&self) -> usize {
        self.tasks
            .iter()
            .map(|task| -> usize {
                match task {
                    Task::Addition(items) => items.iter().sum(),
                    Task::Multiplication(items) => items.iter().product(),
                }
            })
            .sum()
    }
}

#[derive(Debug, Error)]
pub enum Day06Error {
    #[error("could not open file")]
    FileError(#[from] std::io::Error),
    #[error("could not parse input")]
    ParseError(#[from] WorksheetParseError),
}

pub fn day06() -> Result<(), Day06Error> {
    let path = std::path::PathBuf::from("resources/day06.txt");
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let mut string = String::new();
    reader.read_to_string(&mut string)?;
    let worksheet = Worksheet::parse_numbers_rowwise(&string)?;
    let total = worksheet.compute_grand_total();
    println!("day06: with row-numbers, the worksheet total is {total}");
    let worksheet = Worksheet::parse_numbers_colwise(&string)?;
    let total = worksheet.compute_grand_total();
    println!("day06: with column-numbers, the worksheet total is {total}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    static EXAMPLE: &str = "123 328  51 64 
 45 64  387 23 
  6 98  215 314
*   +   *   +  ";

    #[test]
    fn test_input_01() -> anyhow::Result<()> {
        let worksheet = Worksheet::parse_numbers_rowwise(EXAMPLE)?;
        let total = worksheet.compute_grand_total();
        assert_eq!(total, 4277556);
        Ok(())
    }

    #[test]
    fn test_input_02() -> anyhow::Result<()> {
        let worksheet = Worksheet::parse_numbers_colwise(EXAMPLE)?;
        let total = worksheet.compute_grand_total();
        assert_eq!(total, 3263827);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum Rotation {
    Left(u32),
    Right(u32),
}

#[derive(thiserror::Error, Debug)]
enum RotationParseError {
    #[error("expect format '(L|R)<number>', got {0}")]
    WrongFormat(String),
}

impl Rotation {
    fn parse(line: &str) -> Result<Self, RotationParseError> {
        let (dir, count) = line
            .split_at_checked(1)
            .ok_or_else(|| RotationParseError::WrongFormat(line.to_string()))?;
        let count = count
            .parse()
            .map_err(|_| RotationParseError::WrongFormat(line.to_string()))?;
        match dir.to_uppercase().as_str() {
            "L" => Ok(Rotation::Left(count)),
            "R" => Ok(Rotation::Right(count)),
            _ => Err(RotationParseError::WrongFormat(line.to_string())),
        }
    }
}

#[derive(Debug)]
struct Dial {
    position: u32,
}

impl Dial {
    fn rotate(&mut self, rotation: Rotation) -> u32 {
        match rotation {
            Rotation::Left(ticks) => {
                let inverted = (100 - self.position) % 100;
                let rotated = (inverted + ticks) % 100;
                self.position = (100 - rotated) % 100
            }
            Rotation::Right(ticks) => {
                self.position = (self.position + ticks) % 100;
            }
        }
        self.position
    }

    fn rotate_counting_passes_over_zero(&mut self, rotation: Rotation) -> u32 {
        let passes = match rotation {
            Rotation::Left(ticks) => {
                let inverted = (100 - self.position) % 100;
                (inverted + ticks) / 100
            }
            Rotation::Right(ticks) => (self.position + ticks) / 100,
        };
        self.rotate(rotation);
        passes
    }
}

impl Default for Dial {
    fn default() -> Self {
        Self { position: 50 }
    }
}

struct Instructions {
    rotations: Vec<Rotation>,
}

#[derive(thiserror::Error, Debug)]
enum InstructionsParseError {
    #[error("error while reading")]
    ReadError(#[from] std::io::Error),
    #[error("wrong format at line {0}")]
    WrongAtLine(usize, #[source] RotationParseError),
}

impl Instructions {
    fn parse(input: impl std::io::BufRead) -> Result<Instructions, InstructionsParseError> {
        let rotations = input
            .lines()
            .enumerate()
            .map(|(i, line)| {
                Rotation::parse(&line?).map_err(|e| InstructionsParseError::WrongAtLine(i, e))
            })
            .collect::<Result<Vec<_>, InstructionsParseError>>()?;
        Ok(Instructions { rotations })
    }

    fn count_times_pointing_at_zero_exact(&self) -> u32 {
        let mut dial = Dial::default();
        let mut count = 0;
        for &rotation in self.rotations.iter() {
            let after = dial.rotate(rotation);
            if after == 0 {
                count += 1;
            }
        }
        count
    }

    fn count_times_pointing_at_zero_passing(&self) -> u32 {
        let mut dial = Dial::default();
        let mut count = 0;
        for &rotation in self.rotations.iter() {
            count += dial.rotate_counting_passes_over_zero(rotation);
        }
        count
    }
}

pub fn day01() -> anyhow::Result<()> {
    let path = std::path::PathBuf::from("resources/day01.txt");
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let input = Instructions::parse(reader)?;
    let count = input.count_times_pointing_at_zero_exact();
    println!("day01: dial points at zero {count} times.");
    let count = input.count_times_pointing_at_zero_passing();
    println!("day01: dial passes zero {count} times.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    static EXAMPLE: &str = "L68
L30
R48
L5
R60
L55
L1
L99
R14
L82";

    #[test]
    fn test_input_01() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let input = Instructions::parse(cursor)?;
        let count = input.count_times_pointing_at_zero_exact();
        assert_eq!(count, 3);
        Ok(())
    }

    #[test]
    fn test_input_02() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let input = Instructions::parse(cursor)?;
        let count = input.count_times_pointing_at_zero_passing();
        assert_eq!(count, 6);
        Ok(())
    }
}

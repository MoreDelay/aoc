struct BatteryBank {
    batteries: Vec<u8>,
}

#[derive(thiserror::Error, Debug)]
enum BatteryBankParseError {
    #[error("found non-number in joltage string, got {0}")]
    NonNumberInLine(String),
}

impl BatteryBank {
    fn parse(input: &str) -> Result<Self, BatteryBankParseError> {
        let input = input.trim();
        let batteries = input
            .chars()
            .map(|c| {
                c.to_digit(10)
                    .map(|d| d as u8)
                    .ok_or_else(|| BatteryBankParseError::NonNumberInLine(input.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { batteries })
    }

    fn largest_joltage(&self, n_batteries_on: usize) -> Option<usize> {
        if n_batteries_on == 0 {
            return Some(0);
        }

        let n_batteries = self.batteries.len();
        if n_batteries < n_batteries_on {
            return None;
        }

        let mut prev_iter = vec![0; n_batteries];
        let mut this_iter = vec![0; n_batteries];

        for power_of_ten in 0..n_batteries_on {
            this_iter.fill(0);

            let factor = 10usize.pow(power_of_ten as u32);

            let first_relevant = n_batteries_on - 1 - power_of_ten;
            let first_irrelevant = n_batteries - power_of_ten;
            let pos_range = first_relevant..first_irrelevant;

            for pos in pos_range.rev() {
                let cur_value = self.batteries[pos] as usize;
                let prev_recurse = prev_iter.get(pos + 1).copied().unwrap_or(0);
                let this_recurse = this_iter.get(pos + 1).copied().unwrap_or(0);

                let on_take = cur_value * factor + prev_recurse;
                let on_leave = this_recurse;
                this_iter[pos] = on_take.max(on_leave);
            }

            std::mem::swap(&mut prev_iter, &mut this_iter);
        }

        Some(prev_iter[0])
    }
}

struct BatteryBanks(Vec<BatteryBank>);

#[derive(thiserror::Error, Debug)]
enum BatteryBanksParseError {
    #[error("error while reading")]
    ReadError(#[from] std::io::Error),
    #[error("wrong format on line {0}")]
    WrongFormat(usize, #[source] BatteryBankParseError),
}

impl BatteryBanks {
    fn parse(input: impl std::io::BufRead) -> Result<Self, BatteryBanksParseError> {
        let banks = input
            .lines()
            .enumerate()
            .map(|(i, s)| {
                BatteryBank::parse(&s?).map_err(|e| BatteryBanksParseError::WrongFormat(i, e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self(banks))
    }

    fn sum_of_largest_joltages(&self, n_batteries_on: usize) -> usize {
        self.0
            .iter()
            .map(|b| b.largest_joltage(n_batteries_on).unwrap_or(0))
            .sum()
    }
}

pub fn day03() -> anyhow::Result<()> {
    let path = std::path::PathBuf::from("resources/day03.txt");
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let banks = BatteryBanks::parse(reader)?;
    let largest_joltages_with_2 = banks.sum_of_largest_joltages(2);
    println!("day03: sum of largest joltages with 2 batteries is {largest_joltages_with_2}");
    let largest_joltages_with_12 = banks.sum_of_largest_joltages(12);
    println!("day03: sum of largest joltages with 12 batteries is {largest_joltages_with_12}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    static EXAMPLE: &str = "987654321111111
811111111111119
234234234234278
818181911112111";

    #[test]
    fn test_input_01() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let banks = BatteryBanks::parse(cursor)?;
        let sum = banks.sum_of_largest_joltages(2);
        assert_eq!(sum, 357);
        Ok(())
    }

    #[test]
    fn test_input_02() -> anyhow::Result<()> {
        let cursor = Cursor::new(EXAMPLE);
        let banks = BatteryBanks::parse(cursor)?;
        let sum = banks.sum_of_largest_joltages(12);
        assert_eq!(sum, 3121910778619);
        Ok(())
    }
}

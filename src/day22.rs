use anyhow::Result;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use crate::util::{self, AocError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Secret(usize);

impl Secret {
    fn parse(input: &str) -> Result<Self> {
        Ok(Secret(input.parse().map_err(|_| AocError::ParseError)?))
    }

    fn evolve(self) -> Self {
        const MASK: usize = 16777215;
        let Secret(val) = self;

        let temp = val << 6;
        let val = temp ^ val;
        let val = val & MASK;

        let temp = val >> 5;
        let val = temp ^ val;
        let val = val & MASK;

        let temp = val << 11;
        let val = temp ^ val;
        let val = val & MASK;

        Secret(val)
    }

    fn evolve_many(self, times: usize) -> Self {
        let mut s = self;
        for _ in 0..times {
            s = s.evolve();
        }
        s
    }

    fn make_banana_price(&self) -> usize {
        let &Secret(val) = self;
        val % 10
    }
}

fn parse_secrets(input: &str) -> Result<Vec<Secret>> {
    input
        .split("\n")
        .filter(|s| !s.is_empty())
        .map(Secret::parse)
        .collect()
}

fn sum_last_secrets(secrets: &[Secret], times: usize) -> usize {
    secrets
        .iter()
        .map(|s| s.evolve_many(times).0)
        .sum::<usize>()
}

struct ChangeSequence {
    size: usize,
    storage: Vec<isize>,
}

impl ChangeSequence {
    fn new(size: usize) -> Self {
        let storage = Vec::with_capacity(size);
        Self { size, storage }
    }
    fn push(&mut self, value: isize) {
        if self.storage.len() == 4 {
            self.storage.remove(0);
        }
        self.storage.push(value);
    }

    fn get(&self) -> Option<[isize; 4]> {
        if self.storage.len() != self.size {
            None
        } else {
            Some(self.storage.clone().try_into().expect("always size 4"))
        }
    }
}

fn find_best_banana_bargain(secrets: &[Secret], seq_size: usize, changes: usize) -> usize {
    let mut map = HashMap::new();
    for secret in secrets {
        let mut set = HashSet::new();
        let mut sequence = ChangeSequence::new(seq_size);
        let mut prev = *secret;
        let mut cur = secret.evolve();
        for _ in 0..changes {
            let prev_val = prev.make_banana_price();
            let cur_val = cur.make_banana_price();
            let change = match prev_val.cmp(&cur_val) {
                Ordering::Less => -(prev_val.abs_diff(cur_val) as isize),
                Ordering::Equal | Ordering::Greater => prev_val.abs_diff(cur_val) as isize,
            };
            sequence.push(change);

            if let Some(seq) = sequence.get() {
                if !set.contains(&seq) {
                    set.insert(seq);
                    let price = cur.make_banana_price();
                    let overall_profit = map.entry(seq).or_insert(0);
                    *overall_profit += price;
                }
            }

            prev = cur;
            cur = cur.evolve();
        }
    }

    *map.values().max().expect("any sequence valid")
}

pub fn run() -> Result<()> {
    println!("day 22");
    let path = PathBuf::from("./resources/day22.txt");
    let data = util::get_data_string(&path)?;
    let secrets = parse_secrets(&data)?;
    let total = sum_last_secrets(&secrets, 2000);
    println!("sum of evolved secret numbers: {total}");
    let bananas = find_best_banana_bargain(&secrets, 4, 2000);
    println!("best banana bargain: {bananas}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_secrets() {
        let input = "1
10
100
2024";
        let secrets = parse_secrets(input).unwrap();
        assert_eq!(secrets.len(), 4);

        let secrets = secrets
            .iter()
            .map(|s| s.evolve_many(2000))
            .collect::<Vec<_>>();
        assert_eq!(secrets[0], Secret(8685429));
        assert_eq!(secrets[1], Secret(4700978));
        assert_eq!(secrets[2], Secret(15273692));
        assert_eq!(secrets[3], Secret(8667524));

        let total = secrets.iter().map(|Secret(v)| v).sum::<usize>();
        assert_eq!(total, 37327623);
    }

    #[test]
    fn test_secret_evolve() {
        let secret = Secret(123);
        let secret = secret.evolve();
        assert_eq!(secret, Secret(15887950));
        let secret = secret.evolve();
        assert_eq!(secret, Secret(16495136));
        let secret = secret.evolve();
        assert_eq!(secret, Secret(527345));
        let secret = secret.evolve();
        assert_eq!(secret, Secret(704524));
        let secret = secret.evolve();
        assert_eq!(secret, Secret(1553684));
        let secret = secret.evolve();
        assert_eq!(secret, Secret(12683156));
        let secret = secret.evolve();
        assert_eq!(secret, Secret(11100544));
        let secret = secret.evolve();
        assert_eq!(secret, Secret(12249484));
        let secret = secret.evolve();
        assert_eq!(secret, Secret(7753432));
        let secret = secret.evolve();
        assert_eq!(secret, Secret(5908254));
    }

    #[test]
    fn test_example_banana_bargain() {
        let input = "1
2
3
2024";
        let secrets = parse_secrets(input).unwrap();
        let bananas = find_best_banana_bargain(&secrets, 4, 2000);
        assert_eq!(bananas, 23);
    }
}

use anyhow::Result;
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    path::PathBuf,
    rc::{Rc, Weak},
};

use crate::util::{self, AocError};

#[derive(Clone, Debug)]
struct Computer {
    name: String,
    neighbors: Vec<Weak<RefCell<Computer>>>,
}

impl Eq for Computer {}

impl PartialEq for Computer {
    fn eq(&self, other: &Self) -> bool {
        self.name.eq(&other.name)
    }
}

impl std::hash::Hash for Computer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

struct Network {
    computers: Vec<Rc<RefCell<Computer>>>,
}

impl Computer {
    fn has_neighbor(&self, other: &Computer) -> bool {
        if other.neighbors.len() < self.neighbors.len() {
            other
                .neighbors
                .iter()
                .any(|c| *c.upgrade().expect("all neighbors exist").borrow() == *self)
        } else {
            self.neighbors
                .iter()
                .any(|c| *c.upgrade().expect("all neighbors exist").borrow() == *other)
        }
    }
}

impl Network {
    fn parse(input: &str) -> Result<Self> {
        use AocError::ParseError;
        let mut computers = HashMap::new();

        for s in input.split("\n").filter(|s| !s.is_empty()) {
            let (one, two) = s.trim().split_once("-").ok_or(ParseError)?;
            if two.contains("-") {
                return Err(ParseError.into());
            }
            let one = computers.entry(one).or_insert_with(|| {
                Rc::from(RefCell::from(Computer {
                    name: one.to_string(),
                    neighbors: vec![],
                }))
            });
            let one = Rc::downgrade(one);

            let two = computers.entry(two).or_insert_with(|| {
                Rc::from(RefCell::from(Computer {
                    name: two.to_string(),
                    neighbors: vec![],
                }))
            });
            let two = Rc::downgrade(two);

            // add each other as neighbor
            one.upgrade()
                .expect("all computers in hash map")
                .borrow_mut()
                .neighbors
                .push(two.clone());
            two.upgrade()
                .expect("all computers in hash map")
                .borrow_mut()
                .neighbors
                .push(one.clone());
        }

        let mut computers = computers.into_values().collect::<Vec<_>>();

        // make order deterministic
        computers.sort_by(|a, b| {
            let a = a.borrow();
            let b = b.borrow();
            (a.name.cmp(&b.name)).then((a.neighbors.len()).cmp(&b.neighbors.len()))
        });
        Ok(Network { computers })
    }

    fn count_filtered_cliques<F>(&self, filter: F) -> usize
    where
        F: Fn(&Computer) -> bool,
    {
        let mut cliques = Vec::new();
        for computer in self.computers.iter() {
            let computer = computer.borrow();
            if !filter(&computer) {
                continue;
            }

            for first in computer.neighbors.iter() {
                let first = first.upgrade().expect("all computers in hash map");
                let first = first.borrow();

                for second in computer.neighbors.iter() {
                    let second = second.upgrade().expect("all computers in hash map");
                    let second = second.borrow();

                    if first.has_neighbor(&second) {
                        let mut clique = [
                            computer.name.clone(),
                            first.name.clone(),
                            second.name.clone(),
                        ];
                        clique.sort();
                        if !cliques.contains(&clique) {
                            cliques.push(clique);
                        }
                    }
                }
            }
        }
        cliques.len()
    }

    fn get_largest_clique(&self) -> Vec<String> {
        #[derive(Debug)]
        struct TreeNode {
            index: usize,
            clique: Vec<usize>,
            take: bool,
        }

        // computers are Rc so clone is cheap
        let mut sorted = self.computers.clone();
        sorted.sort_by(|a, b| {
            let a = a.borrow();
            let b = b.borrow();
            a.neighbors.len().cmp(&b.neighbors.len())
        });

        let mut best = Vec::new();
        for computer in sorted.iter() {
            let computer = computer.borrow();
            let n_neighbors = computer.neighbors.len();
            if n_neighbors <= best.len() {
                // no computer after has more neighbors because we sorted
                break;
            }

            let mut queue = VecDeque::new();
            queue.push_back(TreeNode {
                index: 0,
                clique: Vec::new(),
                take: true,
            });
            queue.push_back(TreeNode {
                index: 0,
                clique: Vec::new(),
                take: false,
            });

            while let Some(node) = queue.pop_front() {
                let TreeNode {
                    index,
                    mut clique,
                    take,
                } = node;

                let potential_size = n_neighbors - index + 1 + clique.len();
                if index == n_neighbors || best.len() >= potential_size {
                    continue;
                }

                if take {
                    let neighbor = computer.neighbors[index]
                        .upgrade()
                        .expect("stays valid in this call");
                    let neighbor = neighbor.borrow();
                    let not_connected = clique.iter().any(|&c| {
                        !computer.neighbors[c]
                            .upgrade()
                            .expect("stays valid in this call")
                            .borrow()
                            .has_neighbor(&neighbor)
                    });
                    if not_connected {
                        continue;
                    }

                    clique.push(index);

                    // update best clique
                    if clique.len() + 1 > best.len() {
                        best = vec![computer.name.clone()];
                        best.extend(clique.iter().map(|&i| {
                            computer.neighbors[i]
                                .upgrade()
                                .expect("stays valid in this call")
                                .borrow()
                                .name
                                .clone()
                        }));
                    }
                }

                queue.push_back(TreeNode {
                    index: index + 1,
                    clique: clique.clone(),
                    take: true,
                });
                queue.push_back(TreeNode {
                    index: index + 1,
                    clique,
                    take: false,
                });
            }
        }
        best.sort();
        best
    }
}

fn starts_with_t(c: &Computer) -> bool {
    c.name.starts_with("t")
}

pub fn run() -> Result<()> {
    println!("day 23");
    let path = PathBuf::from("./resources/day23.txt");
    let data = util::get_data_string(&path)?;
    let network = Network::parse(&data)?;
    let count = network.count_filtered_cliques(starts_with_t);
    println!("Got {count} cliques");
    let names = network.get_largest_clique();
    let password = names.join(",");
    println!("LAN party password: {password}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tripple_clique() {
        let input = "kh-tc
qp-kh
de-cg
ka-co
yn-aq
qp-ub
cg-tb
vc-aq
tb-ka
wh-tc
yn-cg
kh-ub
ta-co
de-co
tc-td
tb-wq
wh-td
ta-ka
td-qp
aq-cg
wq-ub
ub-vc
de-ta
wq-aq
wq-vc
wh-yn
ka-de
kh-ta
co-tc
wh-qp
tb-vc
td-yn";
        let network = Network::parse(input).unwrap();
        let count = network.count_filtered_cliques(starts_with_t);
        assert_eq!(count, 7);
    }

    #[test]
    fn test_largest_clique() {
        let input = "kh-tc
qp-kh
de-cg
ka-co
yn-aq
qp-ub
cg-tb
vc-aq
tb-ka
wh-tc
yn-cg
kh-ub
ta-co
de-co
tc-td
tb-wq
wh-td
ta-ka
td-qp
aq-cg
wq-ub
ub-vc
de-ta
wq-aq
wq-vc
wh-yn
ka-de
kh-ta
co-tc
wh-qp
tb-vc
td-yn";
        let network = Network::parse(input).unwrap();
        let names = network.get_largest_clique();
        let password = names.join(",");
        assert_eq!(password, "co,de,ka,ta");
    }
}

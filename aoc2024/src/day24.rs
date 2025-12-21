use anyhow::Result;
use std::{ops::ControlFlow, path::PathBuf};

use crate::util::{self, AocError};

#[derive(Copy, Clone, Debug)]
enum Signal {
    Empty,
    Active(bool),
}

#[derive(Clone, Debug)]
struct Wire {
    name: String,
    signal: Signal,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone, Debug)]
enum Operator {
    OR,
    AND,
    XOR,
}

#[derive(Copy, Clone, Debug)]
struct Gate {
    op: Operator,
    input1: usize,
    input2: usize,
    output: usize,
}

#[derive(Clone)]
struct Circuit {
    wires: Vec<Wire>,
    gates: Vec<Gate>,
}

impl From<bool> for Signal {
    fn from(value: bool) -> Self {
        Signal::Active(value)
    }
}

impl Wire {
    fn active(&self) -> bool {
        !matches!(self.signal, Signal::Empty)
    }
}

impl PartialEq for Wire {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Circuit {
    fn parse(input: &str) -> Result<Circuit> {
        use AocError::ParseError;
        let (wires, gates) = input.trim().split_once("\n\n").ok_or(ParseError)?;
        let mut wires = wires
            .split("\n")
            .filter(|s| !s.is_empty())
            .map(|s| {
                let (name, value) = s.split_once(": ").ok_or(ParseError)?;
                let name = name.into();
                let signal = match value {
                    "0" => Signal::Active(false),
                    "1" => Signal::Active(true),
                    _ => return Err(ParseError.into()),
                };
                Ok(Wire { name, signal })
            })
            .collect::<Result<Vec<_>>>()?;

        let gates = gates
            .split("\n")
            .filter(|s| !s.is_empty())
            .map(|s| {
                let s = s.trim();

                let (input1, s) = s.split_once(" ").ok_or(ParseError)?;
                let (op, s) = s.split_once(" ").ok_or(ParseError)?;
                let (input2, output) = s.split_once(" -> ").ok_or(ParseError)?;

                let input1 = match wires.iter().position(|s| s.name == input1) {
                    Some(index) => index,
                    None => {
                        wires.push(Wire {
                            name: input1.into(),
                            signal: Signal::Empty,
                        });
                        wires.len() - 1
                    }
                };

                let input2 = match wires.iter().position(|s| s.name == input2) {
                    Some(index) => index,
                    None => {
                        wires.push(Wire {
                            name: input2.into(),
                            signal: Signal::Empty,
                        });
                        wires.len() - 1
                    }
                };

                let output = match wires.iter().position(|s| s.name == output) {
                    Some(index) => index,
                    None => {
                        wires.push(Wire {
                            name: output.into(),
                            signal: Signal::Empty,
                        });
                        wires.len() - 1
                    }
                };

                let op = match op {
                    "OR" => Operator::OR,
                    "AND" => Operator::AND,
                    "XOR" => Operator::XOR,
                    _ => return Err(ParseError.into()),
                };

                Ok(Gate {
                    op,
                    input1,
                    input2,
                    output,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Circuit { wires, gates })
    }

    fn single_pass(&mut self) -> ControlFlow<(), ()> {
        let mut changed = false;
        for gate in self.gates.iter() {
            let &Gate {
                op,
                input1,
                input2,
                output,
            } = gate;

            let Wire { signal: input1, .. } = &self.wires[input1];
            let input1 = match input1 {
                Signal::Empty => continue,
                Signal::Active(v) => v,
            };
            let Wire { signal: input2, .. } = &self.wires[input2];
            let input2 = match input2 {
                Signal::Empty => continue,
                Signal::Active(v) => v,
            };
            if self.wires[output].active() {
                continue;
            }
            let res = match op {
                Operator::OR => input1 | input2,
                Operator::AND => input1 & input2,
                Operator::XOR => input1 ^ input2,
            };

            self.wires[output].signal = res.into();
            changed |= true;
        }

        match changed {
            true => ControlFlow::Continue(()),
            false => ControlFlow::Break(()),
        }
    }

    fn filter_wires(&self, prefix: char) -> Vec<&Wire> {
        let mut out_wires = self
            .wires
            .iter()
            .filter(|w| w.name.starts_with(prefix))
            .collect::<Vec<_>>();
        out_wires.sort_by(|l, r| r.name.cmp(&l.name));
        out_wires
    }

    fn create_value_from_wires(&self, prefix: char) -> Result<usize> {
        let out_wires = self
            .filter_wires(prefix)
            .into_iter()
            .map(|w| match w.signal {
                Signal::Empty => Err(AocError::ValueError("empty signal as output".into()).into()),
                Signal::Active(v) => Ok(v),
            })
            .collect::<Result<Vec<_>>>()?;
        let value = out_wires
            .into_iter()
            .fold(0, |acc, v| (acc << 1) + v as usize);
        Ok(value)
    }

    fn propagate_signals(mut self) -> Circuit {
        while let ControlFlow::Continue(()) = self.single_pass() {}
        self
    }

    #[allow(dead_code)]
    fn to_graphviz_representation(&self) -> String {
        let comparison = |l: &Wire, r: &Wire| {
            l.name[1..]
                .cmp(&r.name[1..])
                .then(l.name[..1].cmp(&r.name[..1]))
        };

        let mut io_wires = self
            .wires
            .iter()
            .filter(|w| {
                w.name.starts_with('x') || w.name.starts_with('y') || w.name.starts_with('z')
            })
            .cloned()
            .collect::<Vec<_>>();
        io_wires.sort_by(comparison);

        let mut res = String::from("digraph G {\n    rankdir=LR;\n    node [shape=ellipse]\n");

        let left_subgraph = "    subgraph io_wires {\n        node [shape=box];";
        res.push_str(left_subgraph);

        for wire in io_wires.iter() {
            let name = &wire.name;
            res.push_str(&format!("{name}; "));
        }
        let subgraph_end = "\n    }\n";
        res.push_str(subgraph_end);

        for (index, gate) in self.gates.iter().enumerate() {
            let name = match gate.op {
                Operator::OR => format!("OR_{index}"),
                Operator::AND => format!("AND_{index}"),
                Operator::XOR => format!("XOR_{index}"),
            };
            let input1 = &self.wires[gate.input1].name;
            let input2 = &self.wires[gate.input2].name;
            let output = &self.wires[gate.output].name;
            res.push_str(&format!("    {name} [shape=diamond]\n"));
            res.push_str(&format!("    {input1} -> {name}\n"));
            res.push_str(&format!("    {input2} -> {name}\n"));
            res.push_str(&format!("    {name} -> {output}\n"));
        }

        res.push_str("}\n");
        res
    }
}

pub fn run() -> Result<()> {
    println!("day 24");
    let path = PathBuf::from("./resources/day24.txt");
    let data = util::get_data_string(&path)?;
    let circuit = Circuit::parse(&data)?;
    let out = circuit.clone().propagate_signals();
    let out = out.create_value_from_wires('z')?;
    println!("circuit produces {out}");

    // let dot = circuit.to_graphviz_representation();
    // println!("{dot}");
    let swapped = String::from("nvh,z37,qdg,z12,vvf,z19,dck,fgn");
    let mut swapped = swapped.split(",").collect::<Vec<_>>();
    swapped.sort();
    let swapped = swapped.join(",");

    println!("swapping involves wires: {swapped} (solved by hand)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_wires() {
        let input = "x00: 1
x01: 1
x02: 1
y00: 0
y01: 1
y02: 0

x00 AND y00 -> z00
x01 XOR y01 -> z01
x02 OR y02 -> z02";
        let circuit = Circuit::parse(input).unwrap();
        let out = circuit.propagate_signals();
        let out = out.create_value_from_wires('z').unwrap();
        assert_eq!(out, 4);
    }

    #[test]
    fn test_large_wires() {
        let input = "x00: 1
x01: 0
x02: 1
x03: 1
x04: 0
y00: 1
y01: 1
y02: 1
y03: 1
y04: 1

ntg XOR fgs -> mjb
y02 OR x01 -> tnw
kwq OR kpj -> z05
x00 OR x03 -> fst
tgd XOR rvg -> z01
vdt OR tnw -> bfw
bfw AND frj -> z10
ffh OR nrd -> bqk
y00 AND y03 -> djm
y03 OR y00 -> psh
bqk OR frj -> z08
tnw OR fst -> frj
gnj AND tgd -> z11
bfw XOR mjb -> z00
x03 OR x00 -> vdt
gnj AND wpb -> z02
x04 AND y00 -> kjc
djm OR pbm -> qhw
nrd AND vdt -> hwm
kjc AND fst -> rvg
y04 OR y02 -> fgs
y01 AND x02 -> pbm
ntg OR kjc -> kwq
psh XOR fgs -> tgd
qhw XOR tgd -> z09
pbm OR djm -> kpj
x03 XOR y03 -> ffh
x00 XOR y04 -> ntg
bfw OR bqk -> z06
nrd XOR fgs -> wpb
frj XOR qhw -> z04
bqk OR frj -> z07
y03 OR x01 -> nrd
hwm AND bqk -> z03
tgd XOR rvg -> z12
tnw OR pbm -> gnj";
        let circuit = Circuit::parse(input).unwrap();
        let out = circuit.propagate_signals();
        let out = out.create_value_from_wires('z').unwrap();
        assert_eq!(out, 2024);
    }
}

use anyhow::Result;
use std::{fmt::Display, path::PathBuf};

use crate::util::{self, AocError};

#[derive(Clone, Default, Debug)]
struct Registers {
    reg_a: usize,
    reg_b: usize,
    reg_c: usize,
}

impl Display for Registers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("Register A: {}\n", self.reg_a))?;
        f.write_str(&format!("Register B: {}\n", self.reg_b))?;
        f.write_str(&format!("Register C: {}\n", self.reg_c))
    }
}

#[derive(Copy, Clone, Debug)]
enum Instruction {
    Adv(u8), // reg_a / combo_op -> reg_a
    Bxl(u8), // reg_b xor lit_op -> reg_b
    Bst(u8), // combo_op % 8 -> reg_b
    Jnz(u8), // reg_a != 0 => j lit_op
    Bxc(u8), // reg_b ^ reg_c -> reg_b
    Out(u8), // combo_op % 8 -> out
    Bdv(u8), // reg_a / combo_op -> reg_b
    Cdv(u8), // reg_a / combo_op -> reg_c
    Halt,
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Instruction::*;

        let combo = match self {
            Adv(op) => {
                f.write_str("Adv")?;
                op
            }
            Bxl(op) => {
                f.write_str("Bxl")?;
                f.write_str(&format!("({op})"))?;
                return Ok(());
            }
            Bst(op) => {
                f.write_str("Bst")?;
                op
            }
            Jnz(op) => {
                f.write_str("Jnz")?;
                f.write_str(&format!("(A!=0?->{op})"))?;
                return Ok(());
            }
            Bxc(_op) => {
                f.write_str("Bxc")?;
                return Ok(());
            }
            Out(op) => {
                f.write_str("Out")?;
                op
            }
            Bdv(op) => {
                f.write_str("Bdv")?;
                op
            }
            Cdv(op) => {
                f.write_str("Cdv")?;
                op
            }
            Halt => {
                f.write_str("HALT")?;
                return Ok(());
            }
        };

        match combo {
            v @ 0..=3 => f.write_str(&format!("({v})")),
            4 => f.write_str("(A)"),
            5 => f.write_str("(B)"),
            6 => f.write_str("(C)"),
            7 => f.write_str("(RESERVED)"),
            _ => unreachable!(),
        }
    }
}

impl Instruction {
    fn halt(self) -> bool {
        match self {
            Instruction::Halt => true,
            _ => false,
        }
    }

    fn as_opcode(self) -> [u8; 2] {
        use Instruction::*;
        match self {
            Adv(op) => [0, op],
            Bxl(op) => [1, op],
            Bst(op) => [2, op],
            Jnz(op) => [3, op],
            Bxc(op) => [4, op],
            Out(op) => [5, op],
            Bdv(op) => [6, op],
            Cdv(op) => [7, op],
            Halt => panic!("Halt has no opcode"),
        }
    }
}

#[derive(Clone)]
struct Computer {
    registers: Registers,
    instruction_pointer: usize,
    program: Vec<Instruction>,
    output: Vec<u8>,
}

impl Display for Computer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Computer:\n")?;
        f.write_str(&format!("counter: {}\n", self.instruction_pointer))?;
        self.registers.fmt(f)?;
        f.write_str("\n")?;
        if let Some(v) = self.program.get(0) {
            v.fmt(f)?;
        }
        for v in self.program.iter().skip(1) {
            f.write_str("\n")?;
            v.fmt(f)?;
        }
        Ok(())
    }
}

impl Default for Computer {
    fn default() -> Self {
        Computer {
            registers: Registers::default(),
            instruction_pointer: 0,
            program: Vec::new(),
            output: Vec::new(),
        }
    }
}

impl Computer {
    fn parse(input: &str) -> Result<Computer> {
        use AocError::ParseError;

        let (reg_a, input) = input.split_once("\n").ok_or(ParseError)?;
        let (reg_b, input) = input.split_once("\n").ok_or(ParseError)?;
        let (reg_c, input) = input.split_once("\n").ok_or(ParseError)?;
        let (empty, input) = input.split_once("\n").ok_or(ParseError)?;
        if !empty.is_empty() {
            return Err(ParseError.into());
        }

        let program = match input.split_once("\n") {
            Some((_, input)) if !input.is_empty() => return Err(ParseError.into()),
            Some((program, _)) => program,
            None => input,
        };

        let reg_a = reg_a.strip_prefix("Register A: ").ok_or(ParseError)?;
        let reg_a = reg_a.parse()?;
        let reg_b = reg_b.strip_prefix("Register B: ").ok_or(ParseError)?;
        let reg_b = reg_b.parse()?;
        let reg_c = reg_c.strip_prefix("Register C: ").ok_or(ParseError)?;
        let reg_c = reg_c.parse()?;

        let registers = Registers {
            reg_a,
            reg_b,
            reg_c,
        };

        let program = program.strip_prefix("Program: ").ok_or(ParseError)?;
        let program = program
            .split(",")
            .map(|c| {
                if c.len() != 1 {
                    return Err(ParseError.into());
                }
                let v: u8 = c.parse()?;
                Ok(v)
            })
            .collect::<Result<Vec<_>>>()?;
        let program = program
            .chunks(2)
            .flat_map(<&[u8; 2]>::try_from)
            .map(|&[opcode, operand]| {
                use Instruction::*;

                match opcode {
                    0 => Ok(Adv(operand)),
                    1 => Ok(Bxl(operand)),
                    2 => Ok(Bst(operand)),
                    3 => Ok(Jnz(operand)),
                    4 => Ok(Bxc(operand)),
                    5 => Ok(Out(operand)),
                    6 => Ok(Bdv(operand)),
                    7 => Ok(Cdv(operand)),
                    _ => Err(ParseError.into()),
                }
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Computer {
            registers,
            program,
            ..Default::default()
        })
    }

    fn next_instruction(&mut self) -> Instruction {
        let index = self.instruction_pointer / 2;
        self.instruction_pointer += 2;
        if let Some(&res) = self.program.get(index) {
            res
        } else {
            Instruction::Halt
        }
    }

    fn get_combo_value(&self, operand: u8) -> usize {
        match operand {
            v @ 0..=3 => v as usize,
            4 => self.registers.reg_a,
            5 => self.registers.reg_b,
            6 => self.registers.reg_c,
            7 => panic!("reserved combo instruction!"),
            _ => unreachable!(),
        }
    }

    fn do_instruction(&mut self, instruction: Instruction) {
        use Instruction::*;
        match instruction {
            Adv(op) => {
                let val = self.get_combo_value(op);
                let num = self.registers.reg_a;
                let den = 2usize.pow(val as u32);
                self.registers.reg_a = num / den;
            }
            Bxl(op) => {
                let res = self.registers.reg_b ^ op as usize;
                self.registers.reg_b = res;
            }
            Bst(op) => {
                let val = self.get_combo_value(op);
                self.registers.reg_b = val % 8;
            }
            Jnz(op) => {
                let val = self.registers.reg_a;
                if val != 0 {
                    self.instruction_pointer = op as usize;
                }
            }
            Bxc(_op) => {
                let res = self.registers.reg_b ^ self.registers.reg_c;
                self.registers.reg_b = res;
            }
            Out(op) => {
                let val = self.get_combo_value(op);
                self.output.push((val % 8) as u8);
            }
            Bdv(op) => {
                let val = self.get_combo_value(op);
                let num = self.registers.reg_a;
                let den = 2usize.pow(val as u32);
                self.registers.reg_b = num / den;
            }
            Cdv(op) => {
                let val = self.get_combo_value(op);
                let num = self.registers.reg_a;
                let den = 2usize.pow(val as u32);
                self.registers.reg_c = num / den;
            }
            Halt => (),
        }
    }

    fn reset(&mut self, registers: Registers) {
        self.instruction_pointer = 0;
        self.output.truncate(0);
        self.registers = registers;
    }

    fn run(&mut self) -> &Vec<u8> {
        let mut next = self.next_instruction();
        let mut count = 0;
        while !next.halt() {
            assert!(count < 1_000_000);
            count += 1;
            // println!("count: {count}, next: {next}\nstate: {:?}", self.registers);
            self.do_instruction(next);
            next = self.next_instruction();
        }
        &self.output
    }
}

fn accumulate_string(values: &[u8]) -> String {
    values.iter().fold(String::new(), |mut acc, &v| {
        if !acc.is_empty() {
            acc.push(',');
        }
        acc.push_str(&v.to_string());
        acc
    })
}

#[derive(Debug)]
struct DFSNode {
    step: usize,
    next: usize,
    reg_a: usize,
}

fn find_needed_register_value(mut computer: Computer, expected: &[u8]) -> usize {
    let initial = DFSNode {
        step: 0,
        next: 0,
        reg_a: 0,
    };
    let mut stack = vec![initial];

    while let Some(node) = stack.pop() {
        let DFSNode { step, next, reg_a } = node;

        if step == expected.len() {
            // found reg
            return reg_a;
        }

        assert!(stack.len() < expected.len());
        assert!(step <= expected.len());
        assert!(next <= 7);

        let new_reg = (reg_a << 3) + next;
        let mut registers = Registers::default();
        registers.reg_a = new_reg;
        computer.reset(registers);

        let next_produced = computer.run().first();

        let can_try_next_val = next < 7;
        if can_try_next_val {
            // try next value, either immediately or when next step fails
            let cur_node = DFSNode {
                step,
                next: next + 1,
                reg_a,
            };
            stack.push(cur_node);
        }

        let next_expected = expected[expected.len() - 1 - step];
        let correct_output = next_produced.is_some_and(|&out| out == next_expected);
        if correct_output {
            // go to next step
            let for_next_step = DFSNode {
                step: step + 1,
                next: 0,
                reg_a: new_reg,
            };
            stack.push(for_next_step);
        }
    }

    panic!("nothing works");
}

pub fn run() -> Result<()> {
    println!("day 17");
    let path = PathBuf::from("./resources/day17.txt");
    let data = util::get_data_string(&path)?;
    let mut computer = Computer::parse(&data).unwrap();
    let values = computer.run();
    let output = accumulate_string(values);
    println!("computer outputs: {output}");

    let expected_output = computer
        .program
        .iter()
        .flat_map(|i| i.as_opcode())
        .collect::<Vec<u8>>();
    let needed_reg = find_needed_register_value(computer, &expected_output);
    println!("to get identity program use: {needed_reg}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_computer() {
        let input = "Register A: 729
Register B: 0
Register C: 0

Program: 0,1,5,4,3,0";
        let mut computer = Computer::parse(input).unwrap();
        let values = computer.run();
        let output = accumulate_string(values);
        assert_eq!(output, "4,6,3,5,6,3,5,2,1,0");
    }

    #[test]
    fn test_find_register_value() {
        let input = "Register A: 2024
Register B: 0
Register C: 0

Program: 0,3,5,4,3,0";
        let computer = Computer::parse(input).unwrap();
        let expected_output = computer
            .program
            .iter()
            .flat_map(|i| i.as_opcode())
            .collect::<Vec<u8>>();
        let reg = find_needed_register_value(computer, &expected_output);
        assert_eq!(reg, 117440);
    }
}

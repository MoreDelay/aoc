use anyhow::Result;
use std::path::PathBuf;

use crate::util::{self, AocError};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Free {
    count: usize,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct File {
    id: usize,
    count: usize,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Block {
    Free(Free),
    File(File),
}

struct Layout {
    blocks: Vec<Block>,
    size: usize,
}

struct BlockIter<'a> {
    layout: &'a Layout,
    forward_block_index: usize,
    forward_steps_within: usize,
    backward_block_index: usize,
    backward_steps_within: usize,
}

impl<'a> Layout {
    fn iter(&'a self) -> BlockIter<'a> {
        BlockIter {
            layout: self,
            forward_block_index: 0,
            forward_steps_within: 0,
            backward_block_index: self.blocks.len() - 1,
            backward_steps_within: 0,
        }
    }
}

impl<'a> Iterator for BlockIter<'a> {
    type Item = Block;

    fn next(&mut self) -> Option<Self::Item> {
        let n_blocks = self.layout.blocks.len();
        for index in self.forward_block_index..n_blocks {
            let steps = self.forward_steps_within;
            let cur = self.layout.blocks[index];
            let block = match cur {
                b @ Block::Free(Free { count }) if steps < count => Some(b),
                b @ Block::File(File { count, .. }) if steps < count => Some(b),
                _ => None,
            };
            if block.is_none() {
                self.forward_steps_within = 0;
                continue;
            }
            self.forward_block_index = index;
            self.forward_steps_within += 1;
            return block;
        }
        None
    }
}

impl<'a> DoubleEndedIterator for BlockIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        for index in (0..=self.backward_block_index).rev() {
            let steps = self.backward_steps_within;
            let cur = self.layout.blocks[index];
            let block = match cur {
                b @ Block::Free(Free { count }) if steps < count => Some(b),
                b @ Block::File(File { count, .. }) if steps < count => Some(b),
                _ => None,
            };
            if block.is_none() {
                self.backward_steps_within = 0;
                continue;
            }
            self.backward_block_index = index;
            self.backward_steps_within += 1;
            return block;
        }
        None
    }
}

fn is_even(val: usize) -> bool {
    (val & 1) == 0
}

fn get_layout(input: &str) -> Result<Layout> {
    let blocks: Vec<_> = input
        .trim()
        .chars()
        .map(|c| c.to_digit(10).ok_or_else(|| AocError::ParseError))
        .enumerate()
        .map(|(i, d)| {
            let count = d? as usize;
            if is_even(i) {
                let id = i / 2;
                Ok(Block::File(File { id, count }))
            } else {
                Ok(Block::Free(Free { count }))
            }
        })
        .collect::<Result<_>>()?;

    let size = blocks
        .iter()
        .map(|b| match b {
            Block::Free(Free { count }) => count,
            Block::File(File { count, .. }) => count,
        })
        .sum();

    Ok(Layout { blocks, size })
}

fn compute_checksum(layout: &Layout) -> usize {
    let n_blocks = layout.size;
    let mut backward_files_iter = layout
        .iter()
        .rev()
        .enumerate()
        .filter_map(|(i, b)| match b {
            Block::Free(_) => None,
            Block::File(b) => {
                let rev_i = n_blocks - 1 - i;
                Some((rev_i, b))
            }
        });

    let mut last_right_block = n_blocks;
    let mut checksum = 0;
    for (i, b) in layout.iter().enumerate() {
        if i >= last_right_block {
            break;
        }
        match b {
            Block::Free(_) => {
                let (rev_i, File { id, .. }) = backward_files_iter.next().unwrap();
                last_right_block = rev_i;
                if i >= last_right_block {
                    break;
                }
                checksum += i * id;
            }
            Block::File(File { id, .. }) => checksum += i * id,
        }
    }

    checksum
}

pub fn run() -> Result<()> {
    println!("day 09");
    let path = PathBuf::from("./resources/day09.txt");
    let data = util::get_data_string(&path)?;
    let layout = get_layout(&data)?;
    let checksum = compute_checksum(&layout);
    println!("checksum: {checksum}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iterate_layout() {
        let input = "1234";
        let layout = get_layout(input).unwrap();
        let mut iter = layout.iter();
        assert_eq!(iter.next(), Some(Block::File(File { id: 0, count: 1 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { count: 2 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { count: 2 })));
        assert_eq!(iter.next(), Some(Block::File(File { id: 1, count: 3 })));
        assert_eq!(iter.next(), Some(Block::File(File { id: 1, count: 3 })));
        assert_eq!(iter.next(), Some(Block::File(File { id: 1, count: 3 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { count: 4 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { count: 4 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { count: 4 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { count: 4 })));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iterate_layout_rev() {
        let input = "1234";
        let layout = get_layout(input).unwrap();
        let mut iter = layout.iter().rev();
        assert_eq!(iter.next(), Some(Block::Free(Free { count: 4 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { count: 4 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { count: 4 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { count: 4 })));
        assert_eq!(iter.next(), Some(Block::File(File { id: 1, count: 3 })));
        assert_eq!(iter.next(), Some(Block::File(File { id: 1, count: 3 })));
        assert_eq!(iter.next(), Some(Block::File(File { id: 1, count: 3 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { count: 2 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { count: 2 })));
        assert_eq!(iter.next(), Some(Block::File(File { id: 0, count: 1 })));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_example() {
        let input = "2333133121414131402";
        let layout = get_layout(input).unwrap();
        let checksum = compute_checksum(&layout);
        assert_eq!(checksum, 1928);
    }
}

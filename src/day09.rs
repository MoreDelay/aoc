use anyhow::Result;
use std::path::PathBuf;

use crate::util::{self, AocError};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Free {
    start: usize,
    count: usize,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct File {
    id: usize,
    start: usize,
    count: usize,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Block {
    Free(Free),
    File(File),
}

impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Block::Free(free) => f.write_str(&".".repeat(free.count)),
            Block::File(file) => f.write_str(&file.id.to_string().repeat(file.count)),
        }
    }
}

#[derive(Clone)]
struct Layout {
    blocks: Vec<Block>,
    size: usize,
}

impl std::fmt::Debug for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in self.blocks.iter() {
            f.write_str(&format!("{b:?}"))?;
        }
        Ok(())
    }
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
                b @ Block::Free(Free { count, .. }) if steps < count => Some(b),
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
                b @ Block::Free(Free { count, .. }) if steps < count => Some(b),
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
    let mut blocks = Vec::new();
    let mut start = 0;
    for (i, d) in input
        .trim()
        .chars()
        .map(|c| c.to_digit(10).ok_or_else(|| AocError::ParseError))
        .enumerate()
    {
        let count = d? as usize;
        if is_even(i) {
            let id = i / 2;
            blocks.push(Block::File(File { id, start, count }));
        } else {
            blocks.push(Block::Free(Free { start, count }));
        }
        start += count;
    }

    Ok(Layout {
        blocks,
        size: start,
    })
}

fn compute_blockwise_checksum(layout: &Layout) -> usize {
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

fn compute_checksum(layout: &Layout) -> usize {
    let mut checksum = 0;
    for (i, b) in layout.iter().enumerate() {
        match b {
            Block::File(File { id, .. }) => checksum += i * id,
            _ => continue,
        }
    }
    checksum
}

fn compute_filewise_checksum(layout: &Layout) -> usize {
    let mut all_files: Vec<_> = layout
        .blocks
        .iter()
        .filter_map(|b| match *b {
            Block::File(file) => Some(file),
            _ => None,
        })
        .collect();

    let mut all_frees: Vec<_> = layout
        .blocks
        .iter()
        .filter_map(|b| match *b {
            Block::Free(free) => Some(free),
            _ => None,
        })
        .collect();

    for file in all_files.iter_mut().rev() {
        let File {
            id: _,
            start,
            count: file_size,
        } = file;

        let free_index = all_frees.iter().position(
            |Free {
                 start: free_start,
                 count: free_space,
             }| free_start < start && free_space >= file_size,
        );
        let Some(free_index) = free_index else {
            continue;
        };

        let Free {
            start: free_start,
            count: free_space,
        } = &mut all_frees[free_index];

        file.start = *free_start;
        *free_start += *file_size;
        *free_space -= *file_size;
    }

    all_files.sort_by(|left, right| left.start.cmp(&right.start));

    let mut new_blocks = Vec::with_capacity(all_files.len() + all_files.len() - 1);
    let Some(&first) = all_files.get(0) else {
        return 0;
    };
    new_blocks.push(Block::File(first));

    let mut last = first;

    for file in all_files.iter().skip(1) {
        let last_end = last.start + last.count;
        let diff = file.start - last_end;
        let new_free = Block::Free(Free {
            start: last_end,
            count: diff,
        });
        new_blocks.push(new_free);
        new_blocks.push(Block::File(*file));
        last = *file;
    }
    let new_layout = Layout {
        blocks: new_blocks,
        size: last.start + last.count,
    };
    compute_checksum(&new_layout)
}

pub fn run() -> Result<()> {
    println!("day 09");
    let path = PathBuf::from("./resources/day09.txt");
    let data = util::get_data_string(&path)?;
    let layout = get_layout(&data)?;
    let checksum = compute_blockwise_checksum(&layout);
    println!("checksum blockwise: {checksum}");
    let checksum = compute_filewise_checksum(&layout);
    println!("checksum filewise: {checksum}");
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
        assert_eq!(
            iter.next(),
            Some(Block::File(File {
                id: 0,
                start: 0,
                count: 1
            }))
        );
        assert_eq!(iter.next(), Some(Block::Free(Free { start: 1, count: 2 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { start: 1, count: 2 })));
        assert_eq!(
            iter.next(),
            Some(Block::File(File {
                id: 1,
                start: 3,
                count: 3
            }))
        );
        assert_eq!(
            iter.next(),
            Some(Block::File(File {
                id: 1,
                start: 3,
                count: 3
            }))
        );
        assert_eq!(
            iter.next(),
            Some(Block::File(File {
                id: 1,
                start: 3,
                count: 3
            }))
        );
        assert_eq!(iter.next(), Some(Block::Free(Free { start: 6, count: 4 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { start: 6, count: 4 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { start: 6, count: 4 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { start: 6, count: 4 })));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iterate_layout_rev() {
        let input = "1234";
        let layout = get_layout(input).unwrap();
        let mut iter = layout.iter().rev();
        assert_eq!(iter.next(), Some(Block::Free(Free { start: 6, count: 4 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { start: 6, count: 4 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { start: 6, count: 4 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { start: 6, count: 4 })));
        assert_eq!(
            iter.next(),
            Some(Block::File(File {
                id: 1,
                start: 3,
                count: 3
            }))
        );
        assert_eq!(
            iter.next(),
            Some(Block::File(File {
                id: 1,
                start: 3,
                count: 3
            }))
        );
        assert_eq!(
            iter.next(),
            Some(Block::File(File {
                id: 1,
                start: 3,
                count: 3
            }))
        );
        assert_eq!(iter.next(), Some(Block::Free(Free { start: 1, count: 2 })));
        assert_eq!(iter.next(), Some(Block::Free(Free { start: 1, count: 2 })));
        assert_eq!(
            iter.next(),
            Some(Block::File(File {
                id: 0,
                start: 0,
                count: 1
            }))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_blockwise() {
        let input = "2333133121414131402";
        let layout = get_layout(input).unwrap();
        let checksum = compute_blockwise_checksum(&layout);
        assert_eq!(checksum, 1928);
    }

    #[test]
    fn test_filewise() {
        let input = "2333133121414131402";
        let layout = get_layout(input).unwrap();
        let checksum = compute_filewise_checksum(&layout);
        assert_eq!(checksum, 2858);
    }
}

use anyhow::Result;
use std::path::PathBuf;

use crate::util::{self, AocError};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct FileBlock {
    id: usize,
    start: usize,
    count: usize,
}

#[derive(Clone)]
struct Layout {
    blocks: Vec<FileBlock>,
    size: usize,
}

impl std::fmt::Debug for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut last_end = 0;
        for file in self.blocks.iter() {
            let diff = file.start - last_end;
            if diff > 0 {
                let free = ".".to_string().repeat(diff);
                f.write_str(&free)?;
            }
            let row = file.id.to_string().repeat(file.count);
            f.write_str(&row)?;
            last_end = file.start + file.count;
        }
        let diff = self.size - last_end;
        if diff > 0 {
            let free = ".".to_string().repeat(diff);
            f.write_str(&free)?;
        }

        Ok(())
    }
}

fn get_layout(input: &str) -> Result<Layout> {
    let mut blocks = Vec::new();
    let mut last_end = 0;
    let test = input
        .trim()
        .chars()
        .map(|c| {
            let d = c
                .to_digit(10)
                .map(|d| d as usize)
                .ok_or_else(|| AocError::ParseError);
            Ok(d?)
        })
        .collect::<Result<Vec<_>>>()?;
    for (index, &digit) in test.iter().enumerate() {
        if util::is_even(index) {
            let id = index / 2;
            let new_block = FileBlock {
                id,
                start: last_end,
                count: digit,
            };
            blocks.push(new_block);
        } else {
        }
        last_end += digit;
    }

    Ok(Layout {
        blocks,
        size: last_end,
    })
}

fn defragment_blockwise(layout: &Layout) -> Layout {
    let mut file_iter = layout.blocks.iter();
    let mut run_file = match file_iter.next() {
        Some(file) => file,
        None => return layout.clone(),
    };
    let mut new_blocks = Vec::new();

    let mut last_index = 0;
    // iterate over all files in reverse order
    for last_file in layout.blocks.iter().rev() {
        let mut last_file = *last_file;
        if last_file.id < run_file.id {
            break;
        }
        // distribute current file completely
        'dist: while last_file.count > 0 && run_file.id < last_file.id {
            // find next free slot
            let mut space = run_file.start - last_index;
            while space == 0 && run_file.id < last_file.id {
                new_blocks.push(*run_file);
                last_index = run_file.start + run_file.count;
                run_file = match file_iter.next() {
                    Some(file) => file,
                    None => break 'dist,
                };
                space = run_file.start - last_index;
            }

            // insert new file
            let slot = space.min(last_file.count);
            let insert = FileBlock {
                id: last_file.id,
                start: last_index,
                count: slot,
            };
            new_blocks.push(insert);
            last_file.count -= slot;
            last_index += slot;
        }

        if last_file.count > 0 {
            new_blocks.push(last_file);
        }
    }

    let new_layout = Layout {
        blocks: new_blocks,
        size: layout.size,
    };
    new_layout
}

fn compute_checksum(layout: &Layout) -> usize {
    let mut checksum = 0;
    for file in layout.blocks.iter() {
        for step in 0..file.count {
            checksum += file.id * (file.start + step);
        }
    }
    checksum
}

fn defrag_filewise(layout: &Layout) -> Layout {
    let mut new_blocks = layout.blocks.clone();

    for file in layout.blocks.iter().rev() {
        let mut last_index = 0;
        for (index, run_file) in new_blocks.iter().enumerate() {
            let space = run_file.start - last_index;
            if space >= file.count {
                let new_file = FileBlock {
                    id: file.id,
                    start: last_index,
                    count: file.count,
                };
                new_blocks.insert(index, new_file);
                let remove_index = new_blocks.iter().rposition(|f| f.id == file.id).unwrap();
                new_blocks.remove(remove_index);
                break;
            }
            last_index = run_file.start + run_file.count;
        }
    }

    new_blocks.sort_by(|l, r| l.start.cmp(&r.start));

    let new_layout = Layout {
        blocks: new_blocks,
        size: layout.size,
    };
    new_layout
}

pub fn run() -> Result<()> {
    println!("day 09");
    let path = PathBuf::from("./resources/day09.txt");
    let data = util::get_data_string(&path)?;
    let layout = get_layout(&data)?;
    let defrag_block = defragment_blockwise(&layout);
    let checksum_block = compute_checksum(&defrag_block);
    println!("checksum blockwise: {checksum_block}");
    let defrag_file = defrag_filewise(&layout);
    let checksum_file = compute_checksum(&defrag_file);
    println!("checksum filewise: {checksum_file}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blockwise() {
        let input = "2333133121414131402";
        let layout = get_layout(input).unwrap();
        let defrag = defragment_blockwise(&layout);
        let checksum = compute_checksum(&defrag);
        assert_eq!(checksum, 1928);
    }

    #[test]
    fn test_filewise() {
        let input = "2333133121414131402";
        let layout = get_layout(input).unwrap();
        let defrag = defrag_filewise(&layout);
        let checksum = compute_checksum(&defrag);
        assert_eq!(checksum, 2858);
    }
}

use std::{collections::HashMap, io::Read};

use thiserror::Error;

struct ParsedDevice<'a> {
    label: &'a str,
    outputs: Vec<&'a str>,
}

#[derive(Debug, Error)]
pub enum DeviceParseError {
    #[error("expect format '<label>:( <other-device>)+', got '{0}'")]
    WrongFormat(String),
    #[error("the 'out' device can not be a labeled device with outputs, got '{0}'")]
    OutAsLabel(String),
}

impl<'a> ParsedDevice<'a> {
    fn parse(input: &'a str) -> Result<Self, DeviceParseError> {
        let mut iter = input.split(":");
        let label = iter
            .next()
            .ok_or_else(|| DeviceParseError::WrongFormat(input.to_string()))?;
        let outputs = iter
            .next()
            .ok_or_else(|| DeviceParseError::WrongFormat(input.to_string()))?;
        if iter.next().is_some() || outputs.find(':').is_some() || label.find(' ').is_some() {
            return Err(DeviceParseError::WrongFormat(input.to_string()));
        }

        if label == "out" {
            return Err(DeviceParseError::OutAsLabel(input.to_string()));
        }

        let outputs = outputs.trim();
        let outputs = outputs
            .split(" ")
            .filter(|s| !s.trim().is_empty())
            .collect::<Vec<_>>();
        if outputs.is_empty() {
            return Err(DeviceParseError::WrongFormat(input.to_string()));
        }

        Ok(Self { label, outputs })
    }
}

#[derive(Debug, Clone)]
struct Device<'a> {
    label: &'a str,
    outputs: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
struct ParsedNetwork<'a> {
    devices: Vec<Device<'a>>,
}

#[derive(Debug, Error)]
pub enum NetworkParseError {
    #[error("could not parse device on line {0}")]
    ParseError(usize, #[source] DeviceParseError),
    #[error("device '{0}' on line {1} specifies '{2}' as its output, which has no specification")]
    MissingDevice(String, usize, String),
    #[error("a device may only be specified once, found device '{0}' on line {1} and {2}")]
    DeviceFoundTwice(String, usize, usize),
    #[error("no 'out' label found in outputs")]
    MissingOut,
}

impl<'a> ParsedNetwork<'a> {
    fn parse(input: &'a str) -> Result<Self, NetworkParseError> {
        use std::collections::hash_map::Entry;

        let devices = input
            .lines()
            .enumerate()
            .map(|(i, line)| {
                ParsedDevice::parse(line).map_err(|e| NetworkParseError::ParseError(i, e))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut map: HashMap<&str, usize> = HashMap::new();
        for (l, i) in devices.iter().enumerate().map(|(i, d)| (d.label, i)) {
            match map.entry(l) {
                Entry::Occupied(entry) => {
                    return Err(NetworkParseError::DeviceFoundTwice(
                        entry.key().to_string(),
                        *entry.get(),
                        i,
                    ));
                }
                Entry::Vacant(entry) => entry.insert(i),
            };
        }
        let map = map;

        let devices = devices
            .into_iter()
            .enumerate()
            .map(|(i, ParsedDevice { label, outputs })| {
                let outputs = outputs
                    .into_iter()
                    .map(|l| match l == "out" {
                        true => Ok(None),
                        false => Some(map.get(l).copied().ok_or_else(|| {
                            NetworkParseError::MissingDevice(label.to_string(), i, l.to_string())
                        }))
                        .transpose(),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Device { label, outputs })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let out_present = devices
            .iter()
            .flat_map(|d| d.outputs.iter())
            .any(|i| i.is_none());
        if !out_present {
            return Err(NetworkParseError::MissingOut);
        }

        Ok(Self { devices })
    }
}

struct YouNetwork<'a> {
    devices: Vec<Device<'a>>,
    you: usize,
}

#[derive(Debug, Error)]
pub enum YouNetworkError {
    #[error("no 'you' device found")]
    MissingYou,
}

impl<'a> YouNetwork<'a> {
    fn new(network: ParsedNetwork<'a>) -> Result<Self, YouNetworkError> {
        let ParsedNetwork { devices } = network;
        let Some(you) = devices.iter().position(|d| d.label == "you") else {
            return Err(YouNetworkError::MissingYou);
        };
        Ok(Self { devices, you })
    }

    fn unwrap(self) -> ParsedNetwork<'a> {
        let YouNetwork { devices, .. } = self;
        ParsedNetwork { devices }
    }

    fn paths_to_out(&self) -> usize {
        #[derive(Debug)]
        struct DfsState {
            index: usize,
            last_child: Option<usize>,
        }

        let init = DfsState {
            index: self.you,
            last_child: None,
        };
        let mut stack = Vec::from_iter([init]);

        let mut counter = 0;

        const MAX_ITER: usize = 100_000_000;
        let mut iter = 0;
        while let Some(state) = stack.pop() {
            iter += 1;
            assert!(iter < MAX_ITER);

            let DfsState { index, last_child } = state;
            let device = &self.devices[index];
            let child_index = match last_child {
                Some(c) => c + 1,
                None => 0,
            };
            if child_index >= device.outputs.len() {
                continue;
            }

            let now = DfsState {
                index,
                last_child: Some(child_index),
            };
            stack.push(now);

            let Some(child) = device.outputs[child_index] else {
                counter += 1;
                continue;
            };

            let next = DfsState {
                index: child,
                last_child: None,
            };
            stack.push(next);
        }
        counter
    }
}

struct RackNetwork<'a> {
    devices: Vec<Device<'a>>,
    svr: usize,
}

#[derive(Debug, Error)]
pub enum RackNetworkMissingDevice {
    #[error("no 'svr' device found")]
    Svr,
    #[error("no 'dac' device found")]
    Dac,
    #[error("no 'fft' device found")]
    Fft,
}

#[derive(Debug, Error)]
pub enum RackPathError {
    #[error("detected loop of counting paths at device '{0}'")]
    LoopingPaths(String),
}

impl<'a> RackNetwork<'a> {
    fn new(network: ParsedNetwork<'a>) -> Result<Self, RackNetworkMissingDevice> {
        let ParsedNetwork { devices } = network;
        let Some(svr) = devices.iter().position(|d| d.label == "svr") else {
            return Err(RackNetworkMissingDevice::Svr);
        };
        if !devices.iter().any(|d| d.label == "dac") {
            return Err(RackNetworkMissingDevice::Dac);
        };
        if !devices.iter().any(|d| d.label == "fft") {
            return Err(RackNetworkMissingDevice::Fft);
        };

        Ok(Self { devices, svr })
    }

    fn paths_through_dac_and_fft(&self) -> Result<usize, RackPathError> {
        #[derive(Debug, Copy, Clone, Default)]
        struct PathTracker {
            fft_dac_out: usize,
            dac_out: usize,
            fft_out: usize,
            out: usize,
        }

        impl std::ops::Add for PathTracker {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self {
                    fft_dac_out: self.fft_dac_out + rhs.fft_dac_out,
                    dac_out: self.dac_out + rhs.dac_out,
                    fft_out: self.fft_out + rhs.fft_out,
                    out: self.out + rhs.out,
                }
            }
        }
        impl std::ops::AddAssign for PathTracker {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl PathTracker {
            fn at_fft(&mut self) {
                self.fft_out = self.out;
                self.fft_dac_out = self.dac_out;
                self.out = 0;
                self.dac_out = 0;
            }
            fn at_dac(&mut self) {
                self.dac_out = self.out;
                self.fft_dac_out = self.fft_out;
                self.out = 0;
                self.fft_out = 0;
            }
            fn at_out(&mut self) {
                self.out += 1;
            }
            fn test_loop(&self, contains_fft: bool, contains_dac: bool) -> bool {
                let mut loop_check = *self;
                if contains_fft {
                    loop_check.at_fft();
                }
                if contains_dac {
                    loop_check.at_dac();
                }
                loop_check.fft_dac_out > 0
            }
        }

        #[derive(Debug)]
        struct DfsState {
            index: usize,
            last_child: Option<usize>,
            tracker: PathTracker,
            passed_fft: bool,
            passed_dac: bool,
        }

        #[derive(Debug, Clone)]
        enum CacheState {
            Unvisited,
            InProgress,
            LoopDetected {
                contains_fft: bool,
                contains_dac: bool,
            },
            Completed(PathTracker),
        }

        let mut stack = Vec::new();

        let mut cache = vec![CacheState::Unvisited; self.devices.len()];

        let init = DfsState {
            index: self.svr,
            last_child: None,
            tracker: Default::default(),
            passed_fft: false,
            passed_dac: false,
        };
        stack.push(init);
        cache[self.svr] = CacheState::InProgress;

        const MAX_ITER: usize = 1_000_000;
        let mut iter = 0;
        'stack: while let Some(mut state) = stack.pop() {
            let device = &self.devices[state.index];

            let cur_child = match state.last_child {
                Some(c) => c,
                None => {
                    // first time visiting this node in dfs
                    match device.label {
                        "fft" => state.passed_fft = true,
                        "dac" => state.passed_dac = true,
                        _ => (),
                    }
                    0
                }
            };

            let cache_entry = &cache[state.index];

            match cache_entry {
                CacheState::Unvisited => unreachable!(),
                CacheState::InProgress => (),
                CacheState::LoopDetected { .. } => (),
                CacheState::Completed(_) => unreachable!(),
            }

            for child_index in cur_child.. {
                iter += 1;
                assert!(iter < MAX_ITER);

                if child_index >= device.outputs.len() {
                    // finalize result of this node

                    let mut tracker = state.tracker;
                    match device.label {
                        "fft" => tracker.at_fft(),
                        "dac" => tracker.at_dac(),
                        _ => (),
                    }

                    let cache_entry = &mut cache[state.index];
                    match cache_entry {
                        CacheState::Unvisited => unreachable!(),
                        CacheState::InProgress => (),
                        CacheState::LoopDetected {
                            contains_fft,
                            contains_dac,
                        } if state.tracker.test_loop(*contains_fft, *contains_dac) => {
                            return Err(RackPathError::LoopingPaths(device.label.to_string()));
                        }
                        // not a loop that counts, so discard it
                        CacheState::LoopDetected { .. } => (),
                        CacheState::Completed(_) => unreachable!(),
                    }
                    *cache_entry = CacheState::Completed(tracker);
                    continue 'stack;
                }

                let Some(child) = device.outputs[child_index] else {
                    state.tracker.at_out();
                    continue;
                };

                let child_cache = &mut cache[child];
                match child_cache {
                    CacheState::Unvisited => {
                        *child_cache = CacheState::InProgress;
                        state.last_child = Some(child_index);

                        let next = DfsState {
                            index: child,
                            last_child: None,
                            tracker: PathTracker::default(),
                            passed_fft: state.passed_fft,
                            passed_dac: state.passed_dac,
                        };
                        stack.push(state);
                        stack.push(next);
                        continue 'stack;
                    }
                    CacheState::InProgress => {
                        *child_cache = CacheState::LoopDetected {
                            contains_fft: state.passed_fft,
                            contains_dac: state.passed_dac,
                        }
                    }
                    CacheState::LoopDetected { .. } => (),
                    CacheState::Completed(child_tracker) => {
                        state.tracker += *child_tracker;
                    }
                }
            }
        }

        let CacheState::Completed(tracker) = cache[self.svr] else {
            unreachable!();
        };
        Ok(tracker.fft_dac_out)
    }
}

#[derive(Debug, Error)]
pub enum Day11Error {
    #[error("could not read file")]
    ReadError(#[from] std::io::Error),
    #[error("could not parse input")]
    Parsing(#[from] NetworkParseError),
    #[error("could not create you network")]
    YouNetwork(#[from] YouNetworkError),
    #[error("could not create rack network")]
    RackNetwork(#[from] RackNetworkMissingDevice),
    #[error("issue counting paths in rack network")]
    RackPath(#[from] RackPathError),
}

pub fn day11() -> Result<(), Day11Error> {
    let path = std::path::PathBuf::from("resources/day11.txt");
    let mut file = std::fs::File::open(path)?;
    let mut string = String::new();
    file.read_to_string(&mut string)?;
    let network = ParsedNetwork::parse(&string)?;
    let network = YouNetwork::new(network)?;
    let paths = network.paths_to_out();
    println!("day11: there exist {paths} paths from 'you' to 'out'");
    let network = RackNetwork::new(network.unwrap())?;
    let paths = network.paths_through_dac_and_fft()?;
    println!("day11: there exist {paths} paths from 'svr' through 'fft' and 'dac' to 'out'");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    static EXAMPLE1: &str = "aaa: you hhh
you: bbb ccc
bbb: ddd eee
ccc: ddd eee fff
ddd: ggg
eee: out
fff: out
ggg: out
hhh: ccc fff iii
iii: out";

    static EXAMPLE2: &str = "svr: aaa bbb
aaa: fft
fft: ccc
bbb: tty
tty: ccc
ccc: ddd eee
ddd: hub
hub: fff
eee: dac
dac: fff
fff: ggg hhh
ggg: out
hhh: out";

    #[test]
    fn test_input_01() -> anyhow::Result<()> {
        let network = ParsedNetwork::parse(EXAMPLE1)?;
        let network = YouNetwork::new(network)?;
        let paths = network.paths_to_out();
        assert_eq!(paths, 5);
        Ok(())
    }

    #[test]
    fn test_input_02() -> anyhow::Result<()> {
        let network = ParsedNetwork::parse(EXAMPLE2)?;
        let network = RackNetwork::new(network)?;
        let paths = network.paths_through_dac_and_fft()?;
        assert_eq!(paths, 2);
        Ok(())
    }
}

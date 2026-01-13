mod dfs {
    use thiserror::Error;

    use super::Device;
    use super::DeviceIndex;
    use super::Devices;

    pub trait PathTracker: Copy + Default {
        fn on_device(&mut self, device: &Device);
        fn combine(&mut self, other: &Self);
    }

    pub trait PathCountTracker: PathTracker {
        type State: PathTracker;
        fn count(&self) -> usize;
        fn makes_infinite_loop(&self, looped_path_state: &Self::State) -> bool;
    }

    #[derive(Debug)]
    struct DfsStackFrame<S, C>
    where
        C: PathCountTracker<State = S>,
    {
        index: DeviceIndex,
        last_child: Option<usize>,
        counter: C,
        path_state: S,
    }

    #[derive(Debug, Clone)]
    enum CacheState<S, C>
    where
        C: PathCountTracker<State = S>,
    {
        Unvisited,
        InProgress,
        LoopDetected { looped_path_state: S },
        Completed(C),
    }

    #[derive(Debug, Clone)]
    struct Cache<S, C>
    where
        C: PathCountTracker<State = S>,
    {
        inner: Vec<CacheState<S, C>>,
    }

    impl<S, C> std::ops::Index<DeviceIndex> for Cache<S, C>
    where
        C: PathCountTracker<State = S>,
    {
        type Output = CacheState<S, C>;

        fn index(&self, index: DeviceIndex) -> &Self::Output {
            &self.inner[index.0]
        }
    }

    impl<S, C> std::ops::IndexMut<DeviceIndex> for Cache<S, C>
    where
        C: PathCountTracker<State = S>,
    {
        fn index_mut(&mut self, index: DeviceIndex) -> &mut Self::Output {
            &mut self.inner[index.0]
        }
    }

    pub struct DfsPathCounter<'a, S, C>
    where
        C: PathCountTracker<State = S>,
    {
        pub devices: &'a Devices<'a>,
        pub start: DeviceIndex,
        pub phantom: std::marker::PhantomData<C>,
    }

    const DFS_MAX_ITER: usize = 1_000_000;

    #[derive(Debug, Error)]
    pub enum DfsError {
        #[error("required too many iterations, limited to {DFS_MAX_ITER}")]
        TooManyIterations,
        #[error("detected loop while counting paths at device '{0}'")]
        LoopingPaths(String),
    }

    impl<'a, S, C> DfsPathCounter<'a, S, C>
    where
        S: PathTracker + Default,
        C: PathCountTracker<State = S>,
    {
        pub fn count_paths(&self) -> Result<usize, DfsError> {
            let mut stack = Vec::new();
            let cache = vec![CacheState::Unvisited; self.devices.len()];
            let mut cache = Cache { inner: cache };

            let init = DfsStackFrame {
                index: self.start,
                last_child: None,
                counter: C::default(),
                path_state: S::default(),
            };
            stack.push(init);
            cache[self.start] = CacheState::InProgress;

            let mut iter = 0;

            'stack: while let Some(mut frame) = stack.pop() {
                let device = &self.devices[frame.index];

                let child_index = match frame.last_child {
                    Some(c) => c,
                    None => {
                        // first time on this device node
                        frame.path_state.on_device(device);
                        0
                    }
                };

                let cache_entry = &cache[frame.index];
                match cache_entry {
                    CacheState::Unvisited => unreachable!(),
                    CacheState::InProgress => (),
                    CacheState::LoopDetected { .. } => (),
                    CacheState::Completed(_) => unreachable!(),
                }

                for child_index in child_index..device.outputs.len() {
                    iter += 1;
                    if iter > DFS_MAX_ITER {
                        return Err(DfsError::TooManyIterations);
                    }

                    frame.last_child = Some(child_index);
                    let child = device.outputs[child_index];
                    let child_cache = &mut cache[child];
                    match child_cache {
                        CacheState::Unvisited => {
                            *child_cache = CacheState::InProgress;

                            let next = DfsStackFrame {
                                index: child,
                                last_child: None,
                                counter: C::default(),
                                path_state: frame.path_state,
                            };
                            stack.push(frame);
                            stack.push(next);
                            continue 'stack;
                        }
                        CacheState::InProgress => {
                            *child_cache = CacheState::LoopDetected {
                                looped_path_state: frame.path_state,
                            }
                        }
                        CacheState::LoopDetected { looped_path_state } => {
                            looped_path_state.combine(&frame.path_state)
                        }
                        CacheState::Completed(child_counter) => {
                            frame.counter.combine(child_counter);
                        }
                    }
                }

                // finalize result of this node
                frame.counter.on_device(device);

                let cache_entry = &mut cache[frame.index];
                match cache_entry {
                    CacheState::Unvisited => unreachable!(),
                    CacheState::InProgress => (),
                    CacheState::LoopDetected { looped_path_state }
                        if frame.counter.makes_infinite_loop(looped_path_state) =>
                    {
                        return Err(DfsError::LoopingPaths(device.label.to_string()));
                    }
                    // not a loop that counts, so discard it
                    CacheState::LoopDetected { .. } => (),
                    CacheState::Completed(_) => unreachable!(),
                }
                *cache_entry = CacheState::Completed(frame.counter);
            }

            let CacheState::Completed(counter) = cache[self.start] else {
                unreachable!();
            };
            Ok(counter.count())
        }
    }
}

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

#[derive(Debug, Copy, Clone)]
struct DeviceIndex(usize);

#[derive(Debug, Clone)]
struct Device<'a> {
    label: &'a str,
    outputs: Vec<DeviceIndex>,
}

#[derive(Debug, Clone)]
struct Devices<'a>(Vec<Device<'a>>);

impl<'a> Devices<'a> {
    fn iter(&self) -> impl Iterator<Item = &Device<'a>> {
        self.0.iter()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a> std::ops::Index<DeviceIndex> for Devices<'a> {
    type Output = Device<'a>;

    fn index(&self, index: DeviceIndex) -> &Self::Output {
        &self.0[index.0]
    }
}

#[derive(Debug, Clone)]
struct ParsedNetwork<'a> {
    devices: Devices<'a>,
}

#[derive(Debug, Error)]
pub enum NetworkParseError {
    #[error("could not parse device on line {0}")]
    ParseError(usize, #[source] DeviceParseError),
    #[error("device '{0}' on line {1} specifies '{2}' as its output, which has no specification")]
    MissingDevice(String, usize, String),
    #[error("a device may only be specified once, found device '{0}' on line {1} and {2}")]
    DeviceFoundTwice(String, usize, usize),
    #[error("missing 'out' device as output target")]
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
            .chain(std::iter::once(Ok(ParsedDevice {
                label: "out",
                outputs: Vec::new(),
            })))
            .collect::<Result<Vec<_>, _>>()?;
        let output_to_out = devices
            .iter()
            .flat_map(|d| d.outputs.iter())
            .any(|&out| out == "out");
        if !output_to_out {
            return Err(NetworkParseError::MissingOut);
        }

        let mut map: HashMap<&str, DeviceIndex> = HashMap::new();
        for (l, i) in devices.iter().enumerate().map(|(i, d)| (d.label, i)) {
            match map.entry(l) {
                Entry::Occupied(entry) => {
                    return Err(NetworkParseError::DeviceFoundTwice(
                        entry.key().to_string(),
                        entry.get().0,
                        i,
                    ));
                }
                Entry::Vacant(entry) => entry.insert(DeviceIndex(i)),
            };
        }
        let map = map;

        let devices = devices
            .into_iter()
            .enumerate()
            .map(|(i, ParsedDevice { label, outputs })| {
                let outputs = outputs
                    .into_iter()
                    .map(|l| {
                        map.get(l).copied().ok_or_else(|| {
                            NetworkParseError::MissingDevice(label.to_string(), i, l.to_string())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Device { label, outputs })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let devices = Devices(devices);

        Ok(Self { devices })
    }
}

struct YouNetwork<'a> {
    devices: Devices<'a>,
    you: DeviceIndex,
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
        let you = DeviceIndex(you);
        Ok(Self { devices, you })
    }

    fn unwrap(self) -> ParsedNetwork<'a> {
        let YouNetwork { devices, .. } = self;
        ParsedNetwork { devices }
    }
}

#[derive(Debug, Copy, Clone, Default)]
struct YouPathStateTracker {}

impl dfs::PathTracker for YouPathStateTracker {
    fn on_device(&mut self, _: &Device) {}
    fn combine(&mut self, _: &Self) {}
}

#[derive(Debug, Copy, Clone, Default)]
struct YouPathCountTracker {
    out: usize,
}

impl dfs::PathTracker for YouPathCountTracker {
    fn on_device(&mut self, device: &Device) {
        if let "out" = device.label {
            self.out += 1
        }
    }

    fn combine(&mut self, other: &Self) {
        self.out += other.out;
    }
}

impl dfs::PathCountTracker for YouPathCountTracker {
    type State = YouPathStateTracker;

    fn count(&self) -> usize {
        self.out
    }

    fn makes_infinite_loop(&self, _: &Self::State) -> bool {
        true
    }
}

impl<'a> YouNetwork<'a> {
    fn paths_to_out(&self) -> Result<usize, dfs::DfsError> {
        type Dfs<'a> = dfs::DfsPathCounter<'a, YouPathStateTracker, YouPathCountTracker>;

        let dfs = Dfs {
            devices: &self.devices,
            start: self.you,
            phantom: std::marker::PhantomData,
        };

        dfs.count_paths()
    }
}

struct SvrNetwork<'a> {
    devices: Devices<'a>,
    svr: DeviceIndex,
}

#[derive(Debug, Error)]
pub enum SvrNetworkMissingDevice {
    #[error("no 'svr' device found")]
    Svr,
    #[error("no 'dac' device found")]
    Dac,
    #[error("no 'fft' device found")]
    Fft,
}

impl<'a> SvrNetwork<'a> {
    fn new(network: ParsedNetwork<'a>) -> Result<Self, SvrNetworkMissingDevice> {
        let ParsedNetwork { devices } = network;
        let Some(svr) = devices.iter().position(|d| d.label == "svr") else {
            return Err(SvrNetworkMissingDevice::Svr);
        };
        let svr = DeviceIndex(svr);
        if !devices.iter().any(|d| d.label == "dac") {
            return Err(SvrNetworkMissingDevice::Dac);
        };
        if !devices.iter().any(|d| d.label == "fft") {
            return Err(SvrNetworkMissingDevice::Fft);
        };

        Ok(Self { devices, svr })
    }
}

#[derive(Debug, Copy, Clone, Default)]
struct SvrPathStateTracker {
    fft: bool,
    dac: bool,
}

impl dfs::PathTracker for SvrPathStateTracker {
    fn on_device(&mut self, device: &Device) {
        match device.label {
            "fft" => self.fft = true,
            "dac" => self.dac = true,
            _ => (),
        }
    }

    fn combine(&mut self, other: &Self) {
        self.fft |= other.fft;
        self.dac |= other.dac;
    }
}

#[derive(Debug, Copy, Clone, Default)]
struct SvrPathCountTracker {
    fft_dac_out: usize,
    dac_out: usize,
    fft_out: usize,
    out: usize,
}

impl dfs::PathTracker for SvrPathCountTracker {
    fn on_device(&mut self, device: &Device) {
        match device.label {
            "out" => self.on_out(),
            "fft" => self.on_fft(),
            "dac" => self.on_dac(),
            _ => (),
        }
    }

    fn combine(&mut self, other: &Self) {
        self.fft_dac_out += other.fft_dac_out;
        self.dac_out += other.dac_out;
        self.fft_out += other.fft_out;
        self.out += other.out;
    }
}

impl dfs::PathCountTracker for SvrPathCountTracker {
    type State = SvrPathStateTracker;

    fn count(&self) -> usize {
        self.fft_dac_out
    }

    fn makes_infinite_loop(&self, looped_path_state: &Self::State) -> bool {
        let mut loop_count = *self;
        if looped_path_state.fft {
            loop_count.on_fft();
        }
        if looped_path_state.dac {
            loop_count.on_dac();
        }
        loop_count.fft_dac_out > 0
    }
}

impl SvrPathCountTracker {
    fn on_fft(&mut self) {
        self.fft_out += self.out;
        self.fft_dac_out += self.dac_out;
        self.out = 0;
        self.dac_out = 0;
    }
    fn on_dac(&mut self) {
        self.dac_out += self.out;
        self.fft_dac_out += self.fft_out;
        self.out = 0;
        self.fft_out = 0;
    }
    fn on_out(&mut self) {
        self.out += 1;
    }
}

impl<'a> SvrNetwork<'a> {
    fn paths_through_dac_and_fft(&'a self) -> Result<usize, dfs::DfsError> {
        type Dfs<'a> = dfs::DfsPathCounter<'a, SvrPathStateTracker, SvrPathCountTracker>;

        let dfs = Dfs {
            devices: &self.devices,
            start: self.svr,
            phantom: std::marker::PhantomData,
        };

        dfs.count_paths()
    }
}

#[derive(Debug, Error)]
pub enum Day11Error {
    #[error("could not read file")]
    ReadError(#[from] std::io::Error),
    #[error("could not parse input")]
    Parsing(#[from] NetworkParseError),
    #[error("could not create 'you' network")]
    YouNetwork(#[from] YouNetworkError),
    #[error("could not create 'svr' network")]
    SvrNetwork(#[from] SvrNetworkMissingDevice),
}

pub fn day11() -> Result<(), Day11Error> {
    let path = std::path::PathBuf::from("resources/day11.txt");
    let mut file = std::fs::File::open(path)?;
    let mut string = String::new();
    file.read_to_string(&mut string)?;
    let network = ParsedNetwork::parse(&string)?;
    let network = YouNetwork::new(network)?;
    match network.paths_to_out() {
        Ok(paths) => println!("day11: there exist {paths} paths from 'you' to 'out'"),
        Err(e) => println!("day11: error while counting paths in 'you' network: {e}"),
    }
    let network = SvrNetwork::new(network.unwrap())?;
    match network.paths_through_dac_and_fft() {
        Ok(paths) => {
            println!("day11: there exist {paths} paths from 'svr' through 'fft' and 'dac' to 'out'")
        }
        Err(e) => println!("day11: error while counting paths in 'svr' network: {e}"),
    }
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
        let paths = network.paths_to_out()?;
        assert_eq!(paths, 5);
        Ok(())
    }

    #[test]
    fn test_input_02() -> anyhow::Result<()> {
        let network = ParsedNetwork::parse(EXAMPLE2)?;
        let network = SvrNetwork::new(network)?;
        let paths = network.paths_through_dac_and_fft()?;
        assert_eq!(paths, 2);
        Ok(())
    }
}

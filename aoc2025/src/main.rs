mod day01;
mod day02;
mod day03;
mod day04;
mod day05;
mod day06;

fn main() {
    if false {
        if let Err(e) = day01::day01() {
            println!("failed day01: {e:?}");
        }
        if let Err(e) = day02::day02() {
            println!("failed day02: {e:?}");
        }
        if let Err(e) = day03::day03() {
            println!("failed day03: {e:?}");
        }
        if let Err(e) = day04::day04() {
            println!("failed day04: {e:?}");
        }
        if let Err(e) = day05::day05() {
            println!("failed day05: {e:?}");
        }
    }
    if let Err(e) = day06::day06() {
        println!("failed day06: {e:?}");
    }
}

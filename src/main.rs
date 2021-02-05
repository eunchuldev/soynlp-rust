mod lib;
use crate::lib::*;
fn main() {
    Model::default().extract_nouns();
    println!("Hello, world!");
}

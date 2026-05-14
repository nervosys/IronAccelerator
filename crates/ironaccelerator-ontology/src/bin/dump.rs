//! `ironaccelerator-ontology-dump` — emit the built-in ontology as JSON.
//!
//! Usage:
//!   ironaccelerator-ontology-dump               # pretty JSON to stdout
//!   ironaccelerator-ontology-dump --compact     # minified JSON
//!   ironaccelerator-ontology-dump --stats       # one-line entity counts
//!   ironaccelerator-ontology-dump --section X   # X ∈ hardware|workloads|strategies|optimizations|edges
//!   ironaccelerator-ontology-dump --out path.json
//!
//! Intended for agent toolchains that want a frozen snapshot of the knowledge
//! base without pulling the Rust crate in as a dependency.

use ironaccelerator_ontology::Ontology;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut compact = false;
    let mut stats = false;
    let mut section: Option<String> = None;
    let mut out: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--compact" => compact = true,
            "--stats" => stats = true,
            "--section" => {
                i += 1;
                section = args.get(i).cloned();
            }
            "--out" => {
                i += 1;
                out = args.get(i).cloned();
            }
            "-h" | "--help" => {
                print_help();
                return;
            }
            other => {
                eprintln!("unknown arg: {other}");
                print_help();
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let o = Ontology::global();

    if stats {
        let s = o.stats();
        println!(
            "hardware={} workloads={} strategies={} optimizations={} edges={}",
            s.hardware, s.workloads, s.strategies, s.optimizations, s.edges
        );
        return;
    }

    let value: serde_json::Value = match section.as_deref() {
        None => serde_json::to_value(o).expect("ontology serialises"),
        Some("hardware") => serde_json::to_value(&o.hardware).unwrap(),
        Some("workloads") => serde_json::to_value(&o.workloads).unwrap(),
        Some("strategies") => serde_json::to_value(&o.strategies).unwrap(),
        Some("optimizations") => serde_json::to_value(&o.optimizations).unwrap(),
        Some("edges") => serde_json::to_value(&o.edges).unwrap(),
        Some(other) => {
            eprintln!("unknown section: {other}");
            std::process::exit(2);
        }
    };

    let text = if compact {
        serde_json::to_string(&value).unwrap()
    } else {
        serde_json::to_string_pretty(&value).unwrap()
    };

    match out {
        Some(path) => std::fs::write(&path, text).expect("write json file"),
        None => {
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            lock.write_all(text.as_bytes()).unwrap();
            lock.write_all(b"\n").unwrap();
        }
    }
}

fn print_help() {
    eprintln!(
        "ironaccelerator-ontology-dump — emit the built-in ontology as JSON\n\n\
         USAGE:\n  \
           ironaccelerator-ontology-dump [--compact] [--section SECTION] [--out PATH]\n  \
           ironaccelerator-ontology-dump --stats\n\n\
         SECTIONS: hardware | workloads | strategies | optimizations | edges"
    );
}

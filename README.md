# fluffy-penguin

Implementation of the ENT2 genetic algorithm in Rust, because both are awesome =]

[![Build status](https://travis-ci.com/dymayday/fluffy-penguin.svg?branch=master)](https://travis-ci.com/dymayday/fluffy-penguin)

- [fluffy-penguin](#fluffy-penguin)
    - [About](#about)
    - [Requirements](#requirements)
    - [Usage](#usage)
        - [Examples](#examples)
    - [TODO](#todo)



## About

This library use Evolutionary algorithm ([EA](https://en.wikipedia.org/wiki/Evolutionary_algorithm)) to automatically evolve the topology of the artificial neural networks of each individual in a population.

## Requirements

- Rust compiler: See https://www.rust-lang.org or https://doc.rust-lang.org/book/second-edition/ch01-01-installation.html
- [Graphviz](http://www.graphviz.org/): used to export to SVG the rendered artificial neural networks.

## Usage

Because this crate is still a work in progress, it's not yet published on [crates.io](https://crates.io/) so you need to add this to your `Cargo.toml`:

```toml
[dependencies]
fluffy-penguin = { git = "https://github.com/dymayday/fluffy-penguin.git" }
```

### Examples

You can run a simple math equation fitting by running:

```bash
cargo run --example basic --release
```

The MNIST example is still a work in progress at the moment and need some serious work on the score/fitness algorithm, but you can still run it with:

```bash
cargo run --example mnist --release
```

## TODO

- [ ] Implement CMA-ES (Covariance Matrix Adaptation - Evolution Strategy) and its multi-objective optimization (MO-CMA-ES).
- [ ] Add advanced scoring methods (looking into python's scikit-learn library might be a good start).
- [ ] Implement the EANT2 pruning method of unnecessary connections during neuro-evolution.
- [ ] Improve API documentation.
- [ ] Limit the number of concurrent renderer process and put it in a non blocking thread.
- [ ] Fix dependdencies versions when hitting first beta.

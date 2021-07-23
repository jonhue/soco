# Smoothed Online Convex Optimization

## Development

### Testing

Unit and integration test can be run with `cargo test`.

### CI

We use the linter [Clippy](https://github.com/rust-lang/rust-clippy) and the code formatter [rustfmt](https://github.com/rust-lang/rustfmt) which can be run using `cargo clippy` and `cargo fmt`, respectively.

### Python bindings

[Maturin](https://github.com/PyO3/maturin) can be used to build the Python bindings for this crate.

1. create a virtualenv: `python3 -m venv venv`
1. build new bindings: `maturin develop`

Now, the bindings are available as the `soco` package.

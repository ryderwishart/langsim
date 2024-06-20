# langsim
Compare languages even with just a few translation pairs.

## How to install from source

Clone the repository:

```bash
git clone https://github.com/ryderwishart/langsim.git
```

Run `pip install -r requirements-dev.txt` to install the dependencies.

Run `pip install -e .` to install the package in editable mode.

## How to run

Run `python examples/basic_usage.py` to run the basic usage example.

Run `python examples/using_debug_mode.py` to run the basic usage example with debug mode.

## Tests

Run `pytest` to test the code.

> Note: tests are currently failing apparently due to a mismatch with the `hydra-core` version.

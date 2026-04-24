# shrinking

Dedicated to [Nick Higham](https://nhigham.com/).

`shrinking` implements the algorithms from Higham, Strabić, Šego, [*Restoring
Definiteness via Shrinking, with an Application to Correlation Matrices with a
Fixed Block*](https://eprints.maths.manchester.ac.uk/2470/).

The package repairs indefinite covariance and correlation matrices by shrinking
them toward a positive definite target.

The algorithms were created in 2014. The code was written by Vedran Šego, under
the supervision of Nick Higham. In 2026, the code was refactored, restyled, and
adapted for [PyPI](https://pypi.org/) deployment, mostly by OpenAI's Codex
using the GPT-5.4 model, guided and reviewed by Vedran Šego.

## Scope

This implementation focuses on positive definite targets. The paper also treats
positive semidefinite targets, but this package keeps the public API to the
positive definite case in order to keep the implementation and contract
smaller.

## Installation

```bash
pip install shrinking
```

## Public API

The primary API is the snake_case package interface:

### Validation and Helpers

- `check_pos_def`: Check whether a matrix is positive definite.
- `blocks_to_target`: Build the fixed-block target matrix from fixed block
  sizes.

### S(alpha) Helpers

- `s`: Compute `S(alpha)` from one of the supported target specifications.
- `s_with_target`: Compute `S(alpha)` from an explicit target matrix.
- `s_with_difference`: Compute `S(alpha)` from a precomputed target difference.
- `s_with_fixed_blocks`: Compute `S(alpha)` for a fixed-block target.
- `s_with_identity`: Compute `S(alpha)` when the target is the identity matrix.

### Algorithms

- `bisection`: Compute the shrinking parameter by the bisection method.
- `bisection_with_fixed_block`: Run the fixed-block bisection variant.
- `newton`: Compute the shrinking parameter by Newton's method.
- `gep`: Compute the shrinking parameter by solving a generalized eigenvalue
  problem.
- `gep_with_fixed_block`: Run the fixed-block generalized eigenvalue variant.

Each algorithm also has a `_meta` variant with the same name plus `_meta`,
which returns an `AlgorithmResult` containing the shrinking parameter `alpha`
(the same one that the ordinary function returns) and iteration count.

### Compatibility

The legacy API is available under `shrinking.backwards_compatibility`.

The package accepts `numpy.ndarray`, `numpy.matrix`, and plain nested sequence
inputs. Support for `numpy.matrix` is kept for compatibility with older
numerical code; for new code, `numpy.ndarray` is the natural default. Plain
nested sequences are normalized to `numpy.ndarray`, and mixed inputs use array
semantics unless every input is explicitly a `numpy.matrix`.

## Example

```python
import numpy as np

from shrinking import bisection

matrix0 = np.array([[1.0, 1.2], [1.2, 1.0]])
matrix1 = np.identity(2)
alpha = bisection(matrix0, matrix1=matrix1)
print(alpha)
```

## Development

The commands in this section are for a repository checkout, not for a normal
installed package.

Run the test suite from the repository root with:

```bash
./run_tests.sh
```

Locally, the test runner imports the checkout from `src/`. In CI, the workflow
sets `USE_INSTALLED_PACKAGE=1`, so the tests run against the installed package
instead.

Remove repository-generated artifacts with:

```bash
./clean.sh
```

For the full development tool set used in this repository, including coverage
and the optional demo dependencies, install the package with the development
extra:

```bash
pip install ".[dev]"
```

Note: installing the sdist in an isolated build environment may download the
build backend (`hatchling`). For offline checks, preinstall the build backend
and use `--no-build-isolation`.

For an interactive Python session in the repository with the package import
path configured:

```bash
./try_me.sh
```

To run the demo script from the repository through the same wrapper:

```bash
./try_me.sh demo_shrinking.py 17 17
```

The repository also includes a GitHub Actions workflow that installs the
package and runs the test suite on supported Python versions.

## License

See [`LICENSE`](LICENSE).

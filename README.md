# Numpy2AD

A pure Python package for *source-to-source* transformation of Numpy matrix expressions to [reverse-mode](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation), also referred to as *adjoint*, algorithmic differentiation code.

*Algorithmic / Automatic Differentiation* is the state of the art for computing derivatives in many fields of application, e.g. computational engineering, finance, and machine learning (backpropagation). For more details, [this book](https://epubs.siam.org/doi/10.1137/1.9781611972078) is a good entry point to the subject.

# Usage
**Numpy2AD** offers two modes of adjoint code generation.

1. In *Expression Mode* a given Numpy matrix expression string, e.g.
    ```python
    from numpy2ad import transform_expr
    
    print(transform_expr("D = A @ B + C"))
    ```
    is transformed to its reverse-mode differentiation code:
    ```python
    """
    v0 = A @ B
    D = v0 + C
    v0_a = np.zeros_like(v0)
    C_a += D_a
    v0_a += D_a
    B_a += A.T @ v0_a
    A_a += v0_a @ B.T
    """
    ```
    The generated code can be easily pasted into the desired context, e.g. a Jupyter notebook, where the adjoints (partial derivatives) `A_a`, `B_a`, `C_a`, and `D_a` are initialized. Expression mode is best suited for quick scripting and debugging. 

2. In *Function Mode* a given function, e.g.
    ```python
    def mma(A, B, C):
        return A @ B + C
    ```
    is transformed to its adjoint function with modified signature:
    ```python
    from numpy2ad import transform

    print(transform(mma))
    ```
    ```python
    """
    import numpy as np

    def mma_ad(A, B, C, A_a, B_a, C_a, out_a):
        v0 = A @ B
        out = v0 + C
        v0_a = np.zeros_like(v0)
        C_a += out_a
        v0_a += out_a
        B_a += A.T @ v0_a
        A_a += v0_a @ B.T
        return (out, A_a, B_a, C_a)
    """
    ```
    
    The transformed code can also be conveniently exported as a Python module with the argument `out_file=...` for validation and easier integration into your existing workflows.


# Install
Coming soon...
```bash
$ pip install numpy2ad
```

# Supported Matrix Operations

**Numpy2Ad** currently supports a limited but broadly applicable subset of Numpy matrix operations. We define a "matrix" as a two-dimensional `numpy.ndarray`. If not further specified, the operations are also valid for one-dimensional "vectors".

- Matrix **Addition** `C = A + B` and **Subtraction** `C = A - B`
- Matrix (Inner) **Products** `C = A @ B`
    - Exception: Inner products between two vectors `c = a @ b` as they result in a scalar.
- Matrix **Inverse** `B = np.linalg.inv(A)`
- Matrix **Transpose** `B = A.T`
- **Element-wise** product `C = A * B`
    - Note that dimension of `A` and `B` must match. Numpy's broadcasting rules are **not** considered during code generation, e.g. multiplying a scalar variable by a matrix will lead to **incorrect** derivative code.

# Limitations
- The adjoint compiler currently only generates **first order**, **reverse-mode** derivative code. Forward-mode ("tangents") ist not supported.
- The expression must be *aliasing-free*, i.e. `A = A @ B` is not allowed.
- The input code must not contain any control flow (`if ... else`), nested functions, loops, or other non-differentiable subroutines.

# Demo, Tests, and Benchmarks

Check out `/demo/example_problems.ipynb` for a collection of matrix models that **Numpy2AD** was applied to and tested on. A more in-depth user guide can be found in `/demo/tutorial.ipynb`.

Tests of code generation and numerical correctness of derivatives, compared to naive finite differences, can be found in `/tests/`.

Furthermore, there are some simple runtime benchmarks for the *MMA* and *GLS* model in `/benchmarks/`.

# Building the Package from Source
1. Create a conda environment from the main directory
    ```bash
    $ conda env create --file environment.yml
    ...
    $ conda activate numpy2ad
    ```
2. Build and install the package locally 
    ```bash
    $ conda develop .
    ```

- Alternatively, you can also use `venv` and `pip`:
    ```bash
    $ python -m venv env 
    ...
    $ source env/bin/activate
    ...
    $ pip install -r dev_requirements.txt
    ...
    $ pip install -e .
    ```

# Acknowledgements
This package was developed by Nicholas Book in close collaboration with Prof. Uwe Naumann and Simon Märtens of [STCE](https://www.stce.rwth-aachen.de/) at RWTH Aachen University, Germany.

# License
MIT License, see `LICENSE.txt`.

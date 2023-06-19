## Numpy2AD

A python package for source-to-source transformation of Numpy matrix expressions to reverse mode ("adjoint") algorithmic differentiation code.
# Usage
Numpy2AD offers two modes of code transformation.
1. In *Expression Mode* a given Numpy matrix expression string, e.g.
    ```python
    from numpy2ad import transform_expr
    
    print(transform_expr("D = A @ B + C"))
    ```
    is transformed to its reverse mode (adjoint) differentiation code
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
    The generated code can be easily pasted into the desired context where the derivatives `A_a`, `B_a`, `C_a`, and `D_a` are initialized. Expression mode is best suited for quick scripting and debugging. 
2. In *Function Mode* a given function, e.g.
    ```python
    def mma(A, B, C):
        return A @ B + C
    ```
    is transformed to
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
    its adjoint function with modified signature. 

    The transformed code can be conveniently exported as a python module with the argument "out_file=..." for easier integration.
    

# Install
1. (Optional) Create a virtual environment `py -m venv env` and activate it.
2. Install locally using `pip install -e .` (requires `build` package).

# Limitations

# Acknowledgements

# License

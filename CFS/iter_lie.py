# iter_lie_numeric.py
import numpy as np
from itertools import product
from typing import Callable, List, Tuple

def numerical_gradient(f: Callable[[np.ndarray], float],
                       z: np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar function f at point z."""
    z = np.asarray(z, dtype=float)
    n = z.size
    grad = np.zeros(n, dtype=float)
    for i in range(n):
        z_plus = z.copy(); z_minus = z.copy()
        z_plus[i] += eps
        z_minus[i] -= eps
        grad[i] = (f(z_plus) - f(z_minus)) / (2 * eps)
    return grad

def lie_derivative_at_point(f: Callable[[np.ndarray], float],
                            g: Callable[[np.ndarray], np.ndarray],
                            z: np.ndarray,
                            eps: float = 1e-6) -> float:
    """Compute L_g f at point z numerically: (grad f)(z) . g(z)."""
    grad_f = numerical_gradient(f, z, eps=eps)
    g_val = np.asarray(g(z), dtype=float)
    return float(np.dot(grad_f, g_val))

def compose_lie_function(prev_f: Callable[[np.ndarray], float],
                         g: Callable[[np.ndarray], np.ndarray],
                         eps: float = 1e-6) -> Callable[[np.ndarray], float]:
    """Return a function that computes L_g(prev_f)(z) numerically."""
    def new_f(z: np.ndarray) -> float:
        return lie_derivative_at_point(prev_f, g, z, eps=eps)
    return new_f

def iter_lie_numeric(h_func: Callable[[np.ndarray], float],
                     g_funcs: List[Callable[[np.ndarray], np.ndarray]],
                     z0: np.ndarray,
                     Ntrunc: int,
                     eps: float = 1e-6) -> Tuple[List[Callable[[np.ndarray], float]], np.ndarray]:
    """
    Numerical iterative Lie derivatives.

    Parameters
    ----------
    h_func : callable
        Scalar function h(z): R^n -> R.
    g_funcs : list of callables
        Each g_funcs[i] is a callable g_i(z) returning a length-n array.
    z0 : array-like
        Point at which to evaluate the Lie derivatives (returned in `values`).
    Ntrunc : int
        Maximum word length (>=1).
    eps : float
        Step for finite differences.

    Returns
    -------
    funcs : list of callables
        Functions corresponding to each Lie derivative in the same ordering as `values`.
    values : np.ndarray
        Numeric values of these functions evaluated at z0, shape (total_lderiv,).
    """
    if Ntrunc < 1 or not isinstance(Ntrunc, int):
        raise ValueError("Ntrunc must be a positive integer.")

    m = len(g_funcs)  # number of vector fields
    z0 = np.asarray(z0, dtype=float)

    # compute total number of Lie derivatives: m + m^2 + ... + m^Ntrunc
    total = 0
    for k in range(1, Ntrunc + 1):
        total += m ** k
    funcs = []    # list of callables
    values = np.zeros(total, dtype=float)

    # order: all words of length 1, then length 2, ... (lexicographic by product order)
    idx = 0

    # first-order Lie derivatives: L_{g_j} h
    first_order_funcs = []
    for j in range(m):
        g = g_funcs[j]
        f_j = compose_lie_function(h_func, g, eps=eps)
        first_order_funcs.append(f_j)
        funcs.append(f_j)
        values[idx] = f_j(z0)
        idx += 1

    prev_order_funcs = first_order_funcs

    # higher orders
    for k in range(2, Ntrunc + 1):
        current_order_funcs = []
        # For each word of length k, we want L_{g_{i1} g_{i2} ... g_{ik}} h.
        # We produce them by applying L_{g_{i1}} to the function that corresponds to word (i2,...,ik).
        # We'll follow ordering: iterate over product(range(m), repeat=k) in lexicographic order.
        # To avoid recomputing from scratch, build functions by composing g on shorter-word functions.
        # Approach: generate all words of length k using product, then build by iterative composition.
        # However, for simplicity and clarity we build by nested loops using prev_order_funcs:
        # prev_order_funcs contains functions for words of length k-1 in the same ordering.

        # For each g_index from 0..m-1 (this becomes the first symbol in the word),
        # we apply L_{g_index} to each function in prev_order_funcs in order.
        for i in range(m):
            g_i = g_funcs[i]
            for prev_f in prev_order_funcs:
                new_f = compose_lie_function(prev_f, g_i, eps=eps)
                current_order_funcs.append(new_f)
                funcs.append(new_f)
                values[idx] = new_f(z0)
                idx += 1

        prev_order_funcs = current_order_funcs

    return funcs, values

# ---------------------------
# Example usage (toy system from your docstring)
# dx1 = -x1*x2 + x1 * u1  -> g0 for drift: [-x1*x2, x1]
# dx2 = x1*x2 - x2 * u2   -> g0??? (the doc had 3 vector fields in earlier example)
# We will replicate the example with g0, g1, g2 as in your doc:
if __name__ == "__main__":
    # define state functions and vector fields
    def h(z):
        # output function h = x1
        return float(z[0])

    # vector fields as callables: g0, g1, g2 producing arrays of length 2
    def g0(z):
        x1, x2 = z
        return np.array([-x1 * x2, x1], dtype=float)

    def g1(z):
        x1, x2 = z
        return np.array([0.0, 0.0], dtype=float)  # example; adapt to your system

    def g2(z):
        x1, x2 = z
        return np.array([0.0, -x2], dtype=float)

    g_list = [g0, g1, g2]  # m = 3

    z0 = np.array([1/6, 1/6], dtype=float)
    Ntrunc = 3

    funcs, vals = iter_lie_numeric(h, g_list, z0, Ntrunc)
    print("Total Lie derivatives:", vals.size)
    print("Values at z0:\n", vals)

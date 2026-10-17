"""Build and evaluate symbolic vectors and matrices from Python.

The example covers the crate's small symbolic array API: vector/matrix
construction, arithmetic, row-major numeric evaluation, transpose and
determinant. The evaluation point follows the expression's ordered free-symbol
list, and returned NumPy arrays own their data rather than borrowing a Rust
workspace. Dimension mismatches raise a typed configuration exception.
"""

import numpy as np

from rustedscithe import Expr, ExprMatrix, ExprVector, InvalidConfigurationError


def main() -> None:
    # The free-variable order is sorted and shared by all evaluate methods.
    x, y = Expr.symbols(["x", "y"])
    vector = ExprVector([x + y, x * y])
    matrix = ExprMatrix([[x, y], [x * y, x**2]])
    right = ExprMatrix([[Expr.constant(1.0), y], [x, Expr.constant(2.0)]])
    point = [2.0, 3.0]

    vector_value = vector.evaluate(point)
    matrix_value = matrix.evaluate(point)
    product_value = matrix.matvec(vector).evaluate(point)
    matrix_product = matrix.matmul(right).evaluate(point)
    entrywise_product = matrix.hadamard(right).evaluate(point)
    # Hand-computable values verify row-major evaluation and each operation
    # independently of the display formatting below.
    np.testing.assert_array_equal(vector_value, [5.0, 6.0])
    np.testing.assert_array_equal(matrix_value, [[2.0, 3.0], [6.0, 4.0]])
    np.testing.assert_array_equal(product_value, [28.0, 54.0])
    np.testing.assert_array_equal(matrix_product, [[8.0, 12.0], [14.0, 26.0]])
    np.testing.assert_array_equal(entrywise_product, [[2.0, 9.0], [12.0, 8.0]])
    assert matrix.determinant().evaluate(point) == -10.0
    print("vector:", vector_value)
    print("matrix, row-major:", matrix_value)
    print("matrix-vector product:", product_value)
    print("matrix-matrix product:", matrix_product)
    print("entrywise (Hadamard) product:", entrywise_product)
    print("transpose:", matrix.transpose().evaluate(point))
    print("determinant:", matrix.determinant().evaluate(point))

    # Returned arrays own their memory and can be retained independently.
    saved = np.array(matrix_value, copy=True)
    matrix_value[0, 0] = -100.0
    assert saved[0, 0] == 2.0

    # Shape mismatches raise a typed exception before reaching native asserts.
    try:
        matrix.matvec(ExprVector([x]))
    except InvalidConfigurationError as error:
        print("invalid dimensions:", error)
    else:
        raise AssertionError("matrix-vector shape mismatch was not rejected")


if __name__ == "__main__":
    main()

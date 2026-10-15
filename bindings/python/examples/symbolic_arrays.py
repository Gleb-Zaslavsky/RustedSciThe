"""Symbolic vector/matrix operations with owned NumPy evaluation results."""

import numpy as np

from rustedscithe import Expr, ExprMatrix, ExprVector, InvalidConfigurationError


def main() -> None:
    # The free-variable order is sorted and shared by all evaluate methods.
    x, y = Expr.symbols(["x", "y"])
    vector = ExprVector([x + y, x * y])
    matrix = ExprMatrix([[x, y], [x * y, x**2]])
    point = [2.0, 3.0]

    vector_value = vector.evaluate(point)
    matrix_value = matrix.evaluate(point)
    product_value = matrix.matvec(vector).evaluate(point)
    print("vector:", vector_value)
    print("matrix, row-major:", matrix_value)
    print("matrix-vector product:", product_value)
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


if __name__ == "__main__":
    main()

output = np.maximum(0, weights @ inputs + bias)


class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def matmul(self, other):
        if self.cols != other.rows:
            raise ValueError(
                f"Cannot multiply shapes {self.shape} and {other.shape}: "
                f"inner dimensions {self.cols} != {other.rows}"
            )
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])


if __name__ == "__main__":
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    result = a.matmul(b)
    print(f"a @ b = {result.data}")

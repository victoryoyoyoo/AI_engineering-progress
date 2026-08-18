class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.components])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1])) for i in range(self.shape[0])])

    def __repr__(self):
        return f"Matrix({self.rows})"    

if __name__ == "__main__":
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * 3 = {a * 3}")
    print(f"a.dot(b) = {a.dot(b)}")
    print(f"|a| = {a.magnitude():.4f}")
    print(f"cosine_similarity(a, b) = {a.cosine_similarity(b):.4f}")

    weights = Matrix([[0.1, -0.2, 0.3], [0.4, 0.5, -0.1]])
    input_vec = Vector([1.0, 0.5, -0.3])
    print(f"weights @ input_vec = {weights @ input_vec}")
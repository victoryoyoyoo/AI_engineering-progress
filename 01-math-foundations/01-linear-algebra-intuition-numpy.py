import numpy as np

a = np.array([1, 2, 3], dtype = float)
b = np.array([4, 5, 6], dtype = float)

print(f"a + b = {a + b}")
print(f"a dot b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")
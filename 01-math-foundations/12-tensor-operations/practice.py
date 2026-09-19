import numpy as np

x = np.zeros((8, 28, 28))
y = x.reshape(8, 1, 28, 28)
z = x.reshape(8, -1)
print(x.shape, y.shape, z.shape)

b = np.zeros((8, 28, 28, 3))
c = b.transpose(0, 3, 1, 2)
print(b.shape, c.shape)

A = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
print(A + b)
print((A + b).shape)

a = np.array([1, 2, 3]).reshape(-1, 1)
b = np.array([10, 20, 30, 40]).reshape(1, -1)
print(a * b)
print((a * b).shape)
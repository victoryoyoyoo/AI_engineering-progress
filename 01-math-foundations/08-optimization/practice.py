def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]


class GradientDescent:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]


if __name__ == "__main__":
    params = [-1.0, 1.0]
    optimizer = GradientDescent(lr=0.0005)
    for _ in range(5000):
        grads = rosenbrock_gradient(params)
        params = optimizer.step(params, grads)
    print(f"x={params[0]:.6f}, y={params[1]:.6f}, loss={rosenbrock(params):.8f}")

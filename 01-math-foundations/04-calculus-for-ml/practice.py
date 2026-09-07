x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")


# ---------- 2D梯度下降 + 線性迴歸,用數值梯度(numerical_gradient)實作 ----------

import random


def numerical_gradient(f, point, h=1e-7):
    # 梯度=把每個變數各自的偏導數收集成一個向量。
    # 偏導數算法:只動第i個變數一點點,其他固定不動,看輸出變化多少除以動的量。
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient


def f_2d(point):
    x, y = point
    return x ** 2 + y ** 2


def gradient_descent_2d():
    print("=== 2D梯度下降,f(x,y)=x^2+y^2 ===")
    point = [4.0, 3.0]
    lr = 0.1
    for step in range(30):
        grad = numerical_gradient(f_2d, point)
        # 每個維度各自減掉「學習率 * 對應的梯度分量」
        point = [p - lr * g for p, g in zip(point, grad)]
        loss = f_2d(point)
        if step % 5 == 0 or step == 29:
            print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")


def linear_regression():
    # predict -> compute loss -> compute gradient -> update weight,重複很多次
    # loss用MSE(均方誤差):error^2的平均
    # dw、db是loss對w、對b的偏導數(手推公式,不是數值法):
    #   loss = mean((wx+b-y)^2)
    #   dw = mean(2*error*x)
    #   db = mean(2*error)
    print("\n=== 梯度下降訓練線性迴歸 y=wx+b ===")
    random.seed(42)
    w = random.gauss(0, 1)
    b = random.gauss(0, 1)
    lr = 0.01

    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [3.0, 5.0, 7.0, 9.0, 11.0]  # 真實關係是 y = 2x + 1

    for epoch in range(200):
        total_loss = 0
        dw = 0
        db = 0
        for x, y in zip(xs, ys):
            pred = w * x + b
            error = pred - y
            total_loss += error ** 2
            dw += 2 * error * x
            db += 2 * error
        dw /= len(xs)
        db /= len(xs)
        total_loss /= len(xs)
        w -= lr * dw
        b -= lr * db
        if epoch % 40 == 0 or epoch == 199:
            print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

    print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
    print(f"Actual:  y = 2x + 1")


if __name__ == "__main__":
    gradient_descent_2d()
    linear_regression()
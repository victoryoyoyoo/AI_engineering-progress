import math
import random

# Lesson 4:Calculus for ML(微積分)
# 這堂課是 Top-Down + JIT 策略,課程本身標記 Type: Learn(不是 Build 型的課)
# 這份 reference.py 是完整程式碼 + 逐段中文解釋,practice.py 只記錄實際手打的部分

# ---------------------------------------------------------------------------
# Step 1: 用「數值法」算導數(不用公式,直接逼近極限定義)
# ---------------------------------------------------------------------------
def numerical_derivative(f, x, h=1e-7):
    """
    導數定義:f'(x) = lim(h->0) [f(x+h) - f(x)] / h
    這裡用「中央差分」版本:[f(x+h) - f(x-h)] / (2h)
    比單邊差分更準,因為左右兩邊的誤差會互相抵消一部分。
    h 取很小的數字(這裡 1e-7)當作極限的近似,不用真的算極限。
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def f(x):
    return x ** 2


def demo_numerical_derivative():
    print("=== Step 1: 數值導數 vs 解析導數(手推公式) ===")
    for x in [-2, -1, 0, 1, 2]:
        numerical = numerical_derivative(f, x)
        analytical = 2 * x  # f(x)=x^2 的導數公式是 f'(x)=2x,手推出來的
        print(f"x={x:2d}  f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}")


# ---------------------------------------------------------------------------
# Step 2: 偏導數(partial derivative)跟梯度(gradient)
# ---------------------------------------------------------------------------
def numerical_gradient(f, point, h=1e-7):
    """
    梯度 = 把每個變數各自的偏導數收集成一個向量。
    偏導數的算法:只動第 i 個變數一點點,其他變數固定不動,
    看輸出變化多少,除以動的量,就是對這個變數的偏導數。
    對每個維度都做一次,湊成一個向量,就是梯度。
    """
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient


def f_multi(point):
    x, y = point
    return x ** 2 + 3 * x * y + y ** 2
    # 手推偏導數:
    # df/dx = 2x + 3y (把y當常數)
    # df/dy = 3x + 2y (把x當常數)


def demo_gradient():
    print("\n=== Step 2: 偏導數 / 梯度 ===")
    grad = numerical_gradient(f_multi, [1.0, 2.0])
    print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
    print(f"Analytical gradient at (1,2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")


# ---------------------------------------------------------------------------
# Step 3: 梯度下降(Gradient Descent),1D,找 f(x)=x^2 的最小值
# ---------------------------------------------------------------------------
def demo_gradient_descent_1d():
    """
    梯度下降的核心規則:x_new = x_old - learning_rate * 梯度
    梯度指向「往上爬最快」的方向,所以要往反方向走(減掉它)才會下降。
    這裡從 x=5 開始,每一步都往 x=0(最低點)靠近。
    """
    print("\n=== Step 3: 梯度下降,1D,f(x)=x^2 ===")
    x = 5.0
    lr = 0.1
    for step in range(20):
        grad = 2 * x  # f(x)=x^2 的導數是 2x,這裡直接用解析公式(不用數值法)
        x = x - lr * grad
        if step % 5 == 0 or step == 19:
            print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")


# ---------------------------------------------------------------------------
# Step 4: 梯度下降,2D 版本,f(x,y) = x^2 + y^2(碗狀,最低點在原點)
# ---------------------------------------------------------------------------
def f_2d(point):
    x, y = point
    return x ** 2 + y ** 2


def demo_gradient_descent_2d():
    print("\n=== Step 4: 梯度下降,2D,f(x,y)=x^2+y^2 ===")
    point = [4.0, 3.0]
    lr = 0.1
    for step in range(30):
        grad = numerical_gradient(f_2d, point)
        # 每個維度各自減掉「學習率 * 對應的梯度分量」
        point = [p - lr * g for p, g in zip(point, grad)]
        loss = f_2d(point)
        if step % 5 == 0 or step == 29:
            print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")


# ---------------------------------------------------------------------------
# Step 5: 比較數值導數 vs 解析導數,對多種函數
# ---------------------------------------------------------------------------
def demo_compare_derivatives():
    print("\n=== Step 5: 數值 vs 解析導數,多種函數 ===")
    test_functions = [
        ("x^2", lambda x: x ** 2, lambda x: 2 * x),
        ("x^3", lambda x: x ** 3, lambda x: 3 * x ** 2),
        ("sin(x)", lambda x: math.sin(x), lambda x: math.cos(x)),
        ("e^x", lambda x: math.exp(x), lambda x: math.exp(x)),
        ("1/x", lambda x: 1 / x, lambda x: -1 / x ** 2),
    ]

    x = 2.0
    print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
    print("-" * 50)
    for name, fn, dfn in test_functions:
        num = numerical_derivative(fn, x)
        ana = dfn(x)
        err = abs(num - ana)
        print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
    # 誤差都非常小(接近0),證明數值法算出來的導數跟手推公式幾乎一樣準。


# ---------------------------------------------------------------------------
# Step 6: 二階導數矩陣 Hessian(算「彎曲程度」,不只是斜率)
# ---------------------------------------------------------------------------
def hessian_2d(f, x, y, h=1e-5):
    """
    Hessian 矩陣裝的是「二階偏導數」:對每個變數再求一次導數。
    這裡用數值法逼近:
    fxx: 對x求兩次導數的近似 [f(x+h)-2f(x)+f(x-h)] / h^2
    fyy: 對y求兩次導數,同樣邏輯
    fxy: 對x跟y各求一次導數(混合偏導數)
    """
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]


def saddle(x, y):
    return x ** 2 - y ** 2  # 馬鞍面:一個方向往上彎、另一個方向往下彎


def bowl(x, y):
    return x ** 2 + y ** 2  # 碗狀:兩個方向都往上彎(最小值點)


def demo_hessian():
    print("\n=== Step 6: Hessian矩陣(曲率) ===")
    H_saddle = hessian_2d(saddle, 0.0, 0.0)
    H_bowl = hessian_2d(bowl, 0.0, 0.0)
    print(f"Saddle Hessian: {H_saddle}")  # eigenvalue是2跟-2,一正一負 -> 鞍點
    print(f"Bowl Hessian:   {H_bowl}")    # eigenvalue是2跟2,都是正的 -> 最小值點


# ---------------------------------------------------------------------------
# Step 7: 泰勒展開(Taylor series)——用多項式局部逼近一個函數
# ---------------------------------------------------------------------------
def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    """
    泰勒展開公式:f(x0+h) ~= f(x0) + f'(x0)*h + (1/2)*f''(x0)*h^2 + ...
    order=1 只取到一階項(相當於梯度下降在做的線性逼近)
    order=2 取到二階項(相當於牛頓法在做的二次逼近,多考慮了曲率)
    """
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result


def demo_taylor():
    print("\n=== Step 7: 泰勒展開逼近 sin(x) ===")
    x0 = 0.0
    for h in [0.1, 0.5, 1.0, 2.0]:
        true_val = math.sin(h)
        t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
        t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
        print(f"h={h:.1f}  sin(h)={true_val:.4f}  order1={t1:.4f}  order2={t2:.4f}")
    # h越小,逼近越準;h越大(離x0越遠),逼近誤差越大。
    # 這就是為什麼梯度下降要用小步伐(小learning rate):
    # 每一步都在假設「局部線性逼近夠準」,踏太大步這個假設就會失效。


# ---------------------------------------------------------------------------
# Step 8: 完整示範——用梯度下降訓練一個最簡單的線性迴歸 y = wx + b
# ---------------------------------------------------------------------------
def demo_linear_regression():
    """
    這是神經網路訓練迴圈的縮小版:
    predict -> compute loss -> compute gradient -> update weight,重複很多次。
    loss用MSE(均方誤差):error^2的平均。
    dw、db是loss對w、對b的偏導數(手推公式,不是數值法):
      loss = mean((wx+b-y)^2)
      dw = mean(2*error*x)
      db = mean(2*error)
    """
    print("\n=== Step 8: 梯度下降訓練線性迴歸 y=wx+b ===")
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
    demo_numerical_derivative()
    demo_gradient()
    demo_gradient_descent_1d()
    demo_gradient_descent_2d()
    demo_compare_derivatives()
    demo_hessian()
    demo_taylor()
    demo_linear_regression()
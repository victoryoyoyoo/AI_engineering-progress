"""
Lesson 5: Chain Rule & Automatic Differentiation
自動微分引擎(micrograd 風格)——PyTorch/TensorFlow 底層機制的迷你版

核心公式(鏈鎖法則 Chain Rule):
    y = f(g(x))  =>  dy/dx = f'(g(x)) * g'(x)
    每多一層合成函數,就多乘一個「局部導數」

這支程式從最底層的 Value class 開始,逐步蓋出:
1. 會自動記錄運算過程的數字(Value)
2. 用拓撲排序(topological sort)正確做反向傳播的 backward()
3. 用 Value 蓋出神經元(Neuron)/層(Layer)/多層感知器(MLP)
4. 拿 XOR 問題實際訓練一個網路
5. 用數值梯度檢查(gradient checking)驗證引擎正確性
"""

import math
import random


# ============================================================
# Step 1-4: Value class —— 自動微分引擎本體
# ============================================================
class Value:
    """
    包住一個數字,順便記錄:
    - data: 這個數字本身的值
    - grad: dLoss/d(這個數字) —— 反向傳播算完才會有值,預設是0
    - _prev: 這個數字是由哪些 Value 算出來的(小孩節點)
    - _backward: 「如果我知道自己的 grad,要怎麼把 grad 傳給我的小孩」這個動作
    """

    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None  # 預設什麼都不做(輸入變數/葉節點用)
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- 加法:局部導數是1,直接把上游梯度傳給兩個小孩 ---
    # d(a+b)/da = 1, d(a+b)/db = 1
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # += 很關鍵:同一個變數如果被用在兩個地方,梯度要「加總」不能「覆蓋」
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    # --- 乘法:局部導數是「另一個數的值」---
    # d(a*b)/da = b, d(a*b)/db = a
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # --- ReLU:x>0時導數是1(原封不動傳下去),x<=0時導數是0(擋住,不傳) ---
    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # --- 以下都是為了讓 Value 用起來像普通數字一樣自然,順便補齊訓練神經網路需要的運算 ---

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):   # 讓 5 + Value(3) 也能動(反向加法)
        return self + other

    def __rmul__(self, other):   # 讓 5 * Value(3) 也能動
        return self * other

    def __rsub__(self, other):   # 讓 5 - Value(3) 也能動
        return other + (-self)

    # 次方:d(x^n)/dx = n * x^(n-1),高中微積分冪法則,直接照抄
    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')

        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    # 除法:a/b 拆解成 a * b^(-1),借用已經寫好的乘法+次方,梯度自動正確
    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    # exp: d(e^x)/dx = e^x 本身
    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    # log: d(ln x)/dx = 1/x
    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # tanh: d(tanh x)/dx = 1 - tanh(x)^2
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # --- 反向傳播本體:拓撲排序 + 從輸出往回走 ---
    def backward(self):
        # 拓撲排序(topological sort):確保「小孩一定排在爸媽前面」
        # 這樣從後面往前走時,每個節點被處理到的時候,它的 grad 已經被所有爸媽貢獻完畢
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0  # 種子梯度:dy/dy = 1,反向傳播從這裡開始往回乘
        for v in reversed(topo):
            v._backward()


# ============================================================
# Step 5: 用 Value 蓋神經網路 —— Neuron / Layer / MLP
# ============================================================
class Neuron:
    """一個神經元 = tanh(w1*x1 + w2*x2 + ... + b),w跟b都是可訓練的Value"""

    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """一層 = 好幾個平行的神經元"""

    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """多層感知器(Multi-Layer Perceptron) = 好幾層疊起來"""

    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def demo_train_xor():
    """
    用剛蓋好的 MLP 訓練 XOR 問題(這是驗證micrograd能不能真的訓練網路的經典測試)
    XOR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0,這裡用 -1/1 配合 tanh 的輸出範圍
    """
    print("=== Step 5: 用自製 autograd 引擎訓練 XOR ===")
    random.seed(42)
    model = MLP([2, 4, 1])  # 2個輸入 -> 4個隱藏神經元 -> 1個輸出

    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ys = [-1, 1, 1, -1]

    for step in range(100):
        preds = [model(x) for x in xs]
        loss = sum((p - y) ** 2 for p, y in zip(preds, ys))  # MSE(均方誤差)

        for p in model.parameters():
            p.grad = 0.0  # 每次訓練前一定要歸零,否則grad會累加上一輪的
        loss.backward()

        lr = 0.05
        for p in model.parameters():
            p.data -= lr * p.grad  # 梯度下降更新規則,跟Lesson4完全一樣

        if step % 20 == 0:
            print(f"step {step:3d}  loss = {loss.data:.4f}")

    print("\n訓練後的預測結果:")
    for x, y in zip(xs, ys):
        print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}")
    print()


# ============================================================
# Step 6: 梯度檢查(Gradient Checking)—— 驗證 autodiff 正不正確
# ============================================================
def gradient_check(build_expr, x_val, h=1e-7):
    """
    比較「自動微分算出來的梯度」跟「數值法(中央差分)算出來的梯度」
    兩者應該幾乎一樣(誤差 < 1e-5),不一樣就代表 backward 寫錯了
    """
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)  # 跟Lesson4的中央差分公式完全一樣

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff


def demo_gradient_check():
    print("=== Step 6: 梯度檢查 ===")

    def expr(x):
        return (x ** 3 + x * 2 + 1).tanh()

    ad, num, diff = gradient_check(expr, 0.5)
    print(f"Autodiff:   {ad:.8f}")
    print(f"Numerical:  {num:.8f}")
    print(f"Difference: {diff:.2e}  (應該遠小於 1e-5)")
    print()


# ============================================================
# Step 7: 手動驗證一個小計算圖
# ============================================================
def demo_manual_verify():
    print("=== Step 7: 手動驗證計算圖 ===")
    x1 = Value(2.0)
    x2 = Value(3.0)
    a = x1 * x2           # a = 6.0
    b = a + Value(1.0)    # b = 7.0
    y = b.relu()          # y = 7.0 (因為 7 > 0,relu是identity)

    y.backward()

    print(f"y = {y.data}")            # 7.0
    print(f"dy/dx1 = {x1.grad}")      # 3.0 (= x2的值)
    print(f"dy/dx2 = {x2.grad}")      # 2.0 (= x1的值)
    print("手動推導:y = relu(x1*x2+1),因為輸入>0,relu是identity")
    print("         dy/dx1 = x2 = 3, dy/dx2 = x1 = 2 —— 跟程式算出來一致\n")


def demo_more_complex():
    print("=== Use It: 更複雜的表達式 ===")
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    f = (a * b + c).relu()  # relu(2*(-3)+10) = relu(4) = 4

    f.backward()
    print(f"f = {f.data}")
    print(f"df/da = {a.grad}  (應該等於 b = -3.0)")
    print(f"df/db = {b.grad}  (應該等於 a = 2.0)")
    print(f"df/dc = {c.grad}  (應該等於 1.0,因為 c 是直接加上去的)")
    print()
    print("PyTorch對照(概念上,環境沒裝torch所以用註解說明,邏輯完全一樣):")
    print("  x1 = torch.tensor(2.0, requires_grad=True)")
    print("  x2 = torch.tensor(3.0, requires_grad=True)")
    print("  y = torch.relu(x1 * x2 + 1)")
    print("  y.backward()  # x1.grad=3.0, x2.grad=2.0 —— 跟我們的引擎算出同樣結果")
    print()


if __name__ == "__main__":
    demo_manual_verify()
    demo_more_complex()
    demo_gradient_check()
    demo_train_xor()

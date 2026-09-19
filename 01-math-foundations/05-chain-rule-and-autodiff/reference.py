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

整體概念:
    前向傳播(forward)  : 照正常順序算出結果,同時把「這個數字是由哪些數字算出來的」記成一張圖
    反向傳播(backward) : 從最終輸出開始,沿著這張圖往回走,每一站都做同一件事:
                          「這一站的梯度 = 上游傳來的梯度 × 這一站自己的局部導數」
    Value 的每個運算(加、乘、relu...)只需要定義自己的局部導數,連鎖起來就自動算出任何複雜式子的梯度。
"""

import math    # 標準函式庫:exp、log、tanh 用來實作對應的運算
import random  # 標準函式庫:初始化神經元權重、固定亂數種子


# === 🔴 ===
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
        # _backward 是一個「函數」,存起來等反向傳播時才呼叫;lambda: None 是「什麼都不做」的函數
        self._backward = lambda: None  # 預設什麼都不做(輸入變數/葉節點用)
        # 用 set 存小孩節點(不重複);children 預設是空 tuple,代表這個 Value 是最底層輸入
        self._prev = set(children)
        # _op 目前沒有參與任何計算,單純方便 print 或畫計算圖時知道每個節點怎麼來的
        self._op = op  # 記錄是什麼運算產生的(只用來 debug 或畫圖,不影響計算)

    def __repr__(self):
        # print(Value) 時顯示 data 跟 grad,4 位小數
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- 加法:局部導數是1,直接把上游梯度傳給兩個小孩 ---
    # d(a+b)/da = 1, d(a+b)/db = 1
    def __add__(self, other):
        # 如果 other 是一般數字(例如 Value + 5),先包成 Value,這樣後面處理方式統一
        other = other if isinstance(other, Value) else Value(other)
        # out 是新節點,記錄它是由 self 和 other 相加產生的
        out = Value(self.data + other.data, (self, other), '+')

        # 閉包(closure):_backward 在這裡定義,但等 backward() 走到這個節點時才執行,
        # 它可以直接使用外層的 self、other、out
        # _backward 只在反向傳播時才會被呼叫;此時 out.grad 已經由後面的節點填好了
        # self.grad / other.grad 累加的是「上游梯度 × 局部導數」,加法的局部導數是 1,所以直接傳下去
        def _backward():
            # += 很關鍵:同一個變數如果被用在兩個地方,梯度要「加總」不能「覆蓋」
            # 例:z = x + x,x 貢獻兩次,dz/dx 應該是 2
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    # --- 乘法:局部導數是「另一個數的值」---
    # d(a*b)/da = b, d(a*b)/db = a
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # 乘法節點:前向結果是兩個 data 相乘,children 記錄它依賴 self 和 other
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # 連鎖法則:局部導數(另一個數的值)× 上游梯度(out.grad)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # --- ReLU:x>0時導數是1(原封不動傳下去),x<=0時導數是0(擋住,不傳) ---
    def relu(self):
        # ReLU:負數變 0、正數不變;只有一個小孩(self),所以 children 只有 (self,)
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            # (1.0 if out.data > 0 else 0.0) 是局部導數:輸出大於 0 表示輸入為正,梯度通過;否則被擋住
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # --- 以下都是為了讓 Value 用起來像普通數字一樣自然,順便補齊訓練神經網路需要的運算 ---

    def __neg__(self):
        # -x 等於 x * (-1),直接借用乘法,梯度自動正確
        return self * -1

    def __sub__(self, other):
        # a - b 等於 a + (-b),借用加法與取負
        return self + (-other)

    def __radd__(self, other):   # 讓 5 + Value(3) 也能動(反向加法)
        # 「r」代表 right:Python 遇到 5 + Value 時,int 不知道怎麼加 Value,會改呼叫 Value.__radd__
        return self + other

    def __rmul__(self, other):   # 讓 5 * Value(3) 也能動
        return self * other

    def __rsub__(self, other):   # 讓 5 - Value(3) 也能動
        return other + (-self)

    # 次方:d(x^n)/dx = n * x^(n-1),高中微積分冪法則,直接照抄
    def __pow__(self, n):
        # n 只支援一般數字(不是 Value),所以局部導數可以直接算
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
        e = math.exp(self.data)  # 先算好 e^x,前向結果跟反向的局部導數是同一個數
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
        t = math.tanh(self.data)  # tanh 輸出範圍是 (-1, 1),神經元常用的激活函數
        out = Value(t, (self,), 'tanh')

        def _backward():
            # 局部導數可以直接用前向已算好的 t,不用重算
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # --- 反向傳播本體:拓撲排序 + 從輸出往回走 ---
    def backward(self):
        # 拓撲排序(topological sort):確保「小孩一定排在爸媽前面」
        # 這樣從後面往前走時,每個節點被處理到的時候,它的 grad 已經被所有爸媽貢獻完畢
        topo = []          # 排好序的節點清單(小孩在前、爸媽在後)
        visited = set()    # 已經處理過的節點,避免同一個節點走兩次

        def build_topo(v):
            # 深度優先搜尋:先遞迴走完所有小孩,最後才把自己放進清單,所以小孩一定排在自己前面
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0  # 種子梯度:dy/dy = 1,反向傳播從這裡開始往回乘
        # reversed(topo):從輸出(最後面)開始往回走,每個節點呼叫自己的 _backward,把梯度分給小孩
        for v in reversed(topo):
            v._backward()


# === 🟡 ===
# ============================================================
# Step 5: 用 Value 蓋神經網路 —— Neuron / Layer / MLP
# ============================================================
class Neuron:
    """一個神經元 = tanh(w1*x1 + w2*x2 + ... + b),w跟b都是可訓練的Value"""

    def __init__(self, n_inputs):
        # 每個輸入對應一個權重,初始值在 [-1, 1] 之間隨機;每個權重是 Value,才能被追蹤梯度
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)  # 偏差 bias,初始為 0

    def __call__(self, x):
        # __call__ 讓物件可以像函數一樣呼叫:neuron(x)
        # zip 把權重與輸入配對,乘起來再用 sum 加總;sum 第二個參數 self.b 是起始值,
        # 等於「先放 b,再逐項加上 w*x」
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        # 回傳所有可訓練參數(權重們 + bias),訓練時要逐一更新
        return self.w + [self.b]


class Layer:
    """一層 = 好幾個平行的神經元"""

    def __init__(self, n_inputs, n_outputs):
        # n_outputs 個神經元,每個都吃同一批 n_inputs 個輸入
        # 同一層的每個神經元看到相同的輸入,但各有自己的權重,所以會學到不同的東西
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        # 每個神經元各算一個輸出,湊成 list
        return [n(x) for n in self.neurons]

    def parameters(self):
        # 雙層 list comprehension:先走過每個神經元,再走過該神經元的每個參數,攤平成一維 list
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """多層感知器(Multi-Layer Perceptron) = 好幾層疊起來"""

    def __init__(self, sizes):
        # sizes 例如 [2, 4, 1]:2 個輸入 → 4 個隱藏神經元 → 1 個輸出,共兩層
        # 第 i 層吃 sizes[i] 個輸入、吐 sizes[i+1] 個輸出
        # 例:sizes=[2,4,1] 會建出 Layer(2,4) 與 Layer(4,1) 兩層
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        # 前一層的輸出就是下一層的輸入,依序穿過每一層
        for layer in self.layers:
            x = layer(x)
        # 只有一個輸出時直接回傳那個 Value,不要包在 list 裡
        return x[0] if len(x) == 1 else x

    # 把所有層的所有參數攤平成一個 list,訓練迴圈用它逐一清零梯度、更新數值
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def demo_train_xor():
    """
    用剛蓋好的 MLP 訓練 XOR 問題(這是驗證micrograd能不能真的訓練網路的經典測試)
    XOR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0,這裡用 -1/1 配合 tanh 的輸出範圍
    """
    print("=== Step 5: 用自製 autograd 引擎訓練 XOR ===")
    random.seed(42)  # 固定亂數種子,讓結果可重現
    model = MLP([2, 4, 1])  # 2個輸入 -> 4個隱藏神經元 -> 1個輸出

    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # XOR 是「兩個輸入不同才輸出 1」;它無法用一條直線分開,所以需要隱藏層,是測試網路能不能學到非線性的經典題
    ys = [-1, 1, 1, -1]  # 目標值用 -1/1 而不是 0/1,因為 tanh 的輸出範圍是 (-1, 1)

    for step in range(100):
        # 1. 前向傳播:對每筆資料算出預測(Value,已經帶著計算圖)
        preds = [model(x) for x in xs]
        # 2. 損失:每筆的 (預測 - 目標)² 加總
        # 這裡是誤差平方的加總(沒有除以筆數),對訓練效果沒有差別,只差一個固定倍數
        loss = sum((p - y) ** 2 for p, y in zip(preds, ys))  # MSE(均方誤差)

        # 3. 反向傳播前,先把所有參數的梯度清零
        for p in model.parameters():
            p.grad = 0.0  # 每次訓練前一定要歸零,否則grad會累加上一輪的
        loss.backward()   # 自動算出 loss 對每個參數的梯度

        # 4. 更新參數:往梯度反方向走一小步
        # 學習率 0.05:每次只往梯度反方向走一小步,太大會來回震盪、太小會收斂很慢
        lr = 0.05
        for p in model.parameters():
            p.data -= lr * p.grad  # 梯度下降更新規則,跟Lesson4完全一樣

        # 每 20 步印一次 loss,應該一路下降,代表網路真的在學
        if step % 20 == 0:
            print(f"step {step:3d}  loss = {loss.data:.4f}")

    print("\n訓練後的預測結果:")
    for x, y in zip(xs, ys):
        # 預測值應該靠近目標的 -1 或 1
        # 格式 {y:2d}:整數佔 2 格寬,對齊負號;{...:6.3f}:小數佔 6 格、3 位小數
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
    # build_expr 是「吃一個 Value、回傳 Value」的函數,代表要檢查的算式
    # 用 Value 跑一遍算式並反向傳播,x.grad 就是自動微分算出的導數
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad  # 自動微分的結果

    # 數值法:把 x 往正負各動 h,看輸出差多少;這裡只需要 .data(純數字),不需要梯度
    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)  # 跟Lesson4的中央差分公式完全一樣

    # 兩者的差距越接近 0,代表 backward 的局部導數都寫對了
    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff


def demo_gradient_check():
    print("=== Step 6: 梯度檢查 ===")

    def expr(x):
        # 測試用算式:tanh(x³ + 2x + 1),同時用到次方、乘法、加法、tanh 四種運算
        # x ** 3 走 __pow__、x * 2 走 __mul__、+ 1 走 __add__,tanh 走 tanh 方法,四個運算的 backward 全部會被用到
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
    # 建立計算圖:x1、x2 是葉節點,a、b、y 是依序算出來的中間節點與輸出
    a = x1 * x2           # a = 6.0
    b = a + Value(1.0)    # b = 7.0
    y = b.relu()          # y = 7.0 (因為 7 > 0,relu是identity)

    # 反向傳播:從 y 開始,依序經過 relu、加法、乘法,把梯度分給 x1 與 x2
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
    # a*b = -6,加 c 得 4(正數),relu 不擋,所以梯度可以完整通過
    f = (a * b + c).relu()  # relu(2*(-3)+10) = relu(4) = 4

    # c 是直接相加進來的,加法的局部導數是 1,所以 df/dc = 1
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
    print("  y.backward()  # x1.grad=3.0, x2.grad=2.0 —— 跟上面Value引擎算出的結果一致")
    print()


if __name__ == "__main__":
    # 直接執行這個檔案時才會跑;由簡單到複雜依序示範
    demo_manual_verify()
    demo_more_complex()
    demo_gradient_check()
    demo_train_xor()

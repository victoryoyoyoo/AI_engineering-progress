"""
Lesson 8: Optimization
最佳化——訓練神經網路本質上就是在loss地形裡找山谷最低點。這支程式從最陽春的
梯度下降開始,一路加上momentum(動量)、Adam(自適應學習率),
並用Rosenbrock函數(經典最佳化測試函數)實際比較三種方法的收斂速度。

核心主軸:每一種optimizer都是在回答同一個問題——怎麼更快、更穩地走到山谷底部。
"""


# === 🔴 ===
# ---------- Step 1: 測試函數 ----------

def rosenbrock(params):
    """Rosenbrock函數:經典最佳化測試題,最小值在(1,1),
    山谷又窄又彎,很容易找到山谷、但很難沿著山谷走到底"""
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_gradient(params):
    """手推偏導數:對x跟對y各自求導"""
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]


# ---------- Step 2: 梯度下降(Vanilla Gradient Descent) ----------

class GradientDescent:
    """最陽春的最佳化方法:每個參數都往梯度反方向走一步,步伐大小由學習率決定
    w = w - lr * gradient"""

    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]


# === 🟡 ===
# ---------- Step 3: SGD + Momentum ----------

class SGDMomentum:
    """帶動量的梯度下降:把過去的梯度累積成一個「速度」,
    像球滾下山一樣,不會每次遇到小凹凸就停下來重新出發,
    v = beta*v + gradient (beta通常設0.9,代表保留90%過去的速度)
    w = w - lr*v"""

    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]


# ---------- Step 4: Adam ----------

class Adam:
    """自適應學習率:每個權重各自追蹤兩個東西——
    m(一階矩,梯度的移動平均,類似momentum)
    v(二階矩,梯度平方的移動平均,代表這個權重梯度通常有多大)
    除以sqrt(v)是關鍵:梯度常常很大的權重,除以一個大數字,實際步伐變小;
    梯度很小的權重,除以一個小數字,實際步伐變大。等於每個權重都有自己專屬的學習率。
    m_hat/v_hat是偏差修正(bias correction):因為m、v一開始都是0,
    前幾步會偏小,除以(1-beta^t)補回來,t是目前第幾步。"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]
        self.v = [
            self.beta2 * v + (1 - self.beta2) * g ** 2
            for v, g in zip(self.v, grads)
        ]

        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]

        return [
            p - self.lr * mh / (vh ** 0.5 + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]


# ---------- Step 5: 跑起來比較 ----------

def optimize(optimizer, func, grad_func, start, steps=5000):
    """用給定的optimizer跑固定步數,回傳每一步的參數歷史"""
    params = list(start)
    history = [params[:]]
    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history


def demo_compare_optimizers():
    start = [-1.0, 1.0]

    gd_history = optimize(GradientDescent(lr=0.0005), rosenbrock, rosenbrock_gradient, start)
    sgd_history = optimize(SGDMomentum(lr=0.0001, momentum=0.9), rosenbrock, rosenbrock_gradient, start)
    adam_history = optimize(Adam(lr=0.01), rosenbrock, rosenbrock_gradient, start)

    print("Rosenbrock函數最佳化比較(最小值在x=1, y=1, loss=0):")
    for name, history in [("GD", gd_history), ("SGD+M", sgd_history), ("Adam", adam_history)]:
        final = history[-1]
        loss = rosenbrock(final)
        print(f"{name:6s} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")


if __name__ == "__main__":
    demo_compare_optimizers()
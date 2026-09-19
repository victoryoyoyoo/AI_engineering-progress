"""
Lesson 8: Optimization
最佳化——訓練神經網路本質上就是在loss地形裡找山谷最低點。這支程式從最陽春的
梯度下降開始,一路加上momentum(動量)、Adam(自適應學習率),
並用Rosenbrock函數(經典最佳化測試函數)實際比較三種方法的收斂速度。

核心主軸:每一種optimizer都是在回答同一個問題——怎麼更快、更穩地走到山谷底部。

檔案結構:
    🔴 Rosenbrock測試函數 + 最陽春的梯度下降
    🟡 SGD+Momentum、Adam、跑起來比較
所有optimizer都用同一個介面:step(params, grads) 吃進目前參數跟梯度,回傳更新後的參數。
參數用純Python的list表示(不用NumPy),方便逐項對照公式。
"""


# === 🔴 ===
# ---------- Step 1: 測試函數 ----------

def rosenbrock(params):
    """Rosenbrock函數:經典最佳化測試題,最小值在(1,1),
    山谷又窄又彎,很容易找到山谷、但很難沿著山谷走到底"""
    # 把長度2的list拆成x、y兩個變數(tuple unpacking),比params[0]、params[1]好讀
    x, y = params
    # 公式 f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    # 兩項都是平方,所以f>=0;只有x=1且y=1時兩項同時為0,f=0是全域最小值
    # 第二項的係數100讓山谷兩側非常陡:y稍微偏離x^2,loss就暴增,形成又窄又彎的山谷
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_gradient(params):
    """手推偏導數:對x跟對y各自求導"""
    x, y = params
    # 對x偏微分,用連鎖律(Lesson 5):
    #   第一項 (1-x)^2      -> 2*(1-x)*(-1)              = -2*(1-x)
    #   第二項 100*(y-x^2)^2 -> 100*2*(y-x^2)*(-2x)       = 200*(y-x^2)*(-2x)
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    # 對y偏微分:第一項跟y無關,微分為0;第二項 -> 100*2*(y-x^2)*1 = 200*(y-x^2)
    df_dy = 200 * (y - x ** 2)
    # 回傳list,順序跟params一致:[對x的偏導, 對y的偏導],梯度就是這兩個數字組成的向量
    return [df_dx, df_dy]


# ---------- Step 2: 梯度下降(Vanilla Gradient Descent) ----------

class GradientDescent:
    """最陽春的最佳化方法:每個參數都往梯度反方向走一步,步伐大小由學習率決定
    w = w - lr * gradient"""

    def __init__(self, lr=0.001):
        # lr(learning rate,學習率)是唯一的超參數:太大會震盪甚至發散,太小會走很慢
        self.lr = lr

    def step(self, params, grads):
        # 梯度指向「往上爬最快」的方向,所以要減掉(往反方向走)才是下山
        # zip(params, grads)把兩個list同一位置的值配成一對,list comprehension逐對計算
        # 例:params=[-1,1]、grads=[4,0]、lr=0.5 -> [-1-0.5*4, 1-0.5*0] = [-3, 1]
        # 不改動原本的params,而是回傳新的list(沒有副作用,方便記錄每一步的歷史)
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
        # momentum就是公式裡的beta:0代表完全沒有慣性(退化成一般梯度下降),越接近1慣性越大
        self.momentum = momentum
        # 速度要跟參數個數一樣長,但建構時還不知道有幾個參數,所以先設None,第一次step才建立
        self.velocity = None

    def step(self, params, grads):
        # 第一次呼叫:速度從0開始(球一開始是靜止的),長度跟參數一樣
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        # 每個參數各自更新速度:新速度 = 90%的舊速度 + 這一步的梯度
        # 方向一致的梯度會一路累加、越滾越快;方向來回震盪的梯度會互相抵銷
        # (這正是momentum能抑制窄山谷裡左右震盪的原因)
        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]
        # 用「速度」而不是「當下梯度」來更新參數
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
        # beta1:一階矩(方向)的衰減率,0.9約等於平均最近10步
        self.beta1 = beta1
        # beta2:二階矩(幅度)的衰減率,0.999約等於平均最近1000步,比beta1記得更久
        self.beta2 = beta2
        # epsilon:加在分母避免除以0(數值穩定性,Lesson 13會細講),1e-8是慣用值
        self.epsilon = epsilon
        # m、v跟velocity一樣,等第一次step才知道要開幾格
        self.m = None
        self.v = None
        # t:目前是第幾步,偏差修正要用,從0開始、每次step先加1
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        # 先加1再用,所以第一步的t=1(若t=0,1-beta^0=0會除以0)
        self.t += 1

        # 一階矩:指數移動平均,新值 = 90%舊值 + 10%新梯度
        # (1-beta1)是新梯度的權重,跟舊值權重加起來剛好是1
        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]
        # 二階矩:同樣的移動平均,但平均的是梯度的平方(g ** 2是次方運算)
        # 平方後永遠是正的,只反映「這個權重的梯度通常有多大」,不管正負
        self.v = [
            self.beta2 * v + (1 - self.beta2) * g ** 2
            for v, g in zip(self.v, grads)
        ]

        # 偏差修正:m、v從0起步,前幾步被拉低。例:t=1時m=0.1*g,除以(1-0.9)=0.1補回g
        # 步數越多,beta^t越接近0,分母越接近1,修正自然消失
        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]

        # 更新量 = lr * m_hat / (sqrt(v_hat) + epsilon)
        # vh ** 0.5 是開根號的寫法;分母是這個權重梯度的典型大小,
        # 所以每個權重的實際步伐大致都是lr的量級,不論它的梯度本身多大多小
        return [
            p - self.lr * mh / (vh ** 0.5 + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]


# ---------- Step 5: 跑起來比較 ----------

def optimize(optimizer, func, grad_func, start, steps=5000):
    """用給定的optimizer跑固定步數,回傳每一步的參數歷史"""
    # list(start)複製一份,避免改到呼叫端傳進來的起點
    params = list(start)
    # params[:]是切片複製;若直接append(params)存的會是同一個物件,之後被覆蓋
    history = [params[:]]
    # 迴圈變數用底線_表示不需要用到它(只是重複steps次)
    for _ in range(steps):
        # 流程固定三步:算梯度 -> 交給optimizer更新 -> 記錄新位置
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history


def demo_compare_optimizers():
    # 起點(-1,1):離最小值(1,1)有一段距離,而且在山谷外側,很考驗optimizer
    start = [-1.0, 1.0]

    # 三種方法用各自合適的學習率(Rosenbrock山谷很陡,GD和momentum的lr要設小,否則會發散)
    gd_history = optimize(GradientDescent(lr=0.0005), rosenbrock, rosenbrock_gradient, start)
    sgd_history = optimize(SGDMomentum(lr=0.0001, momentum=0.9), rosenbrock, rosenbrock_gradient, start)
    adam_history = optimize(Adam(lr=0.01), rosenbrock, rosenbrock_gradient, start)

    print("Rosenbrock函數最佳化比較(最小值在x=1, y=1, loss=0):")
    # 對一串(名稱, 歷史)的tuple做迴圈,同時解構成name、history兩個變數
    for name, history in [("GD", gd_history), ("SGD+M", sgd_history), ("Adam", adam_history)]:
        # history[-1]取最後一項,也就是跑完所有步數後的位置
        final = history[-1]
        loss = rosenbrock(final)
        # f-string格式:{name:6s}是字串佔6格寬;{x:.6f}是小數點後6位
        print(f"{name:6s} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")


# 只有直接執行這支檔案時才跑demo;被別的檔案import時不會執行
if __name__ == "__main__":
    demo_compare_optimizers()

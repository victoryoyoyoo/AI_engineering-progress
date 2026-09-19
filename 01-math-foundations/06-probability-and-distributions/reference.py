"""
Lesson 6: Probability and Distributions
機率與分布——涵蓋機率公理、PMF/PDF、期望值/變異數、抽樣、
softmax/log-softmax/cross-entropy、中央極限定理(CLT) demo

核心主軸:AI系統本質上都在跟不確定性打交道——分類模型輸出的是機率
(softmax),訓練用的loss是機率的期望值(cross-entropy),權重初始化
靠機率分布(normal),這支程式從機率論最底層的公理開始,一路連到
訓練神經網路每一步實際在做的事。

名詞速記:
    PMF(機率質量函數) : 離散變數(像骰子點數)每個結果的機率,加總 = 1
    PDF(機率密度函數) : 連續變數(像身高)某一點的「密度」,不是機率;要對一段區間積分才是機率
    期望值 E[X]         : 長期平均會落在哪裡
    變異數 Var(X)       : 結果離平均值平均有多分散
    logits              : 模型輸出、還沒轉成機率的原始分數,可正可負
"""

import math    # 標準函式庫:exp、log、sqrt、cos、pi
import random  # 標準函式庫:random.random()產生 [0,1) 均勻亂數,所有抽樣函式都從它出發


# === 🔴 ===
# ---------- Step 3: 期望值 / 變異數 ----------

def expected_value(values, probabilities):
    """期望值 E[X]:每個結果乘上自己的機率再加總,機率加權平均"""
    # 例:公平骰子 values=[1..6]、每個機率 1/6 → 1/6*(1+2+...+6) = 3.5
    # zip 把「結果」與「該結果的機率」配成一對;sum(... for ...) 是 generator 寫法,逐項加總
    return sum(v * p for v, p in zip(values, probabilities))


def variance(values, probabilities):
    """變異數 Var(X) = E[(X-mu)^2]:每個結果離平均值的差距平方,再取機率加權平均
    (跟展開版 E[X^2]-(E[X])^2 數學上等價,這裡用定義版,邏輯較直觀)"""
    mu = expected_value(values, probabilities)  # 先算平均值 mu
    # 差距平方:平方讓正負差距都變正、且遠離平均的點被放大;再乘機率做加權平均
    # 公平骰子的答案是 35/12 ≈ 2.9167
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))


# ---------- Step 5: softmax / log-softmax / cross-entropy ----------

def softmax(logits):
    """把原始分數(logits)轉成合法機率分布(全部在0~1之間,加總=1)。
    先減掉最大值再取exp,是數值穩定技巧:避免exp(大數字)直接溢位,
    因為分子分母同時除以同一個常數,比例完全不變"""
    max_logit = max(logits)  # 找出最大分數;減掉它之後最大的一項變成 0,exp(0)=1,不會爆掉
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]  # exp 讓所有數字變正,且分數差距被放大
    total = sum(exps)
    # 每項除以總和,加總剛好等於 1。例:logits [2.0, 1.0, 0.1] → 約 [0.659, 0.242, 0.099]
    return [e / total for e in exps]


def log_softmax(logits):
    """直接算log機率,不用先softmax再取log(避免機率太小時log(0)出現負無窮)。
    用log-sum-exp技巧把取指數跟取log合併簡化成一步"""
    # 推導:log(softmax(z_i)) = z_i - log(sum(exp(z_j))),右邊那項叫 log-sum-exp
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    # log-sum-exp 穩定寫法:max + log(sum(exp(z - max))),數學上等價,但不會溢位
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]


def cross_entropy_loss(logits, target_index):
    """分類問題的loss:正確類別的log機率取負號。
    模型對正確答案越沒信心(機率越低),loss越大,值域是[0,+∞)"""
    # target_index 是正確答案的類別編號;只取正確類別那一項的 log 機率再加負號
    # 機率 1 → loss 0(完全正確且有信心);機率越接近 0 → loss 越大
    log_probs = log_softmax(logits)
    return -log_probs[target_index]


# === 🟡 ===
# ---------- Step 1: 機率基礎 ----------

def factorial(n):
    """n! = n * (n-1) * ... * 1,用來算排列組合"""
    result = 1
    # range(2, n+1) 從 2 乘到 n(乘 1 沒有意義,所以從 2 開始);n=0 或 1 時迴圈不執行,回傳 1
    for i in range(2, n + 1):
        result *= i
    return result


def combinations(n, k):
    """從n個東西裡選k個,不管順序,C(n,k) = n! / (k! * (n-k)!)"""
    # // 是整數除法(結果一定整除,用它避免出現 10.0 這種浮點數)
    # 例:C(5,2) = 120 / (2*6) = 10
    return factorial(n) // (factorial(k) * factorial(n - k))


def conditional_probability(p_a_and_b, p_b):
    """P(A|B) = P(A且B) / P(B) —— 已知B發生的前提下,A發生的機率"""
    # 直覺:把宇宙縮小到「B 已發生」的範圍,看 A 佔其中多少比例
    return p_a_and_b / p_b


# ---------- Step 2: PMF / PDF ----------

def bernoulli_pmf(k, p):
    """伯努利分布:k=1(成功)的機率是p,k=0(失敗)的機率是1-p"""
    # 像丟一次不公平硬幣;條件運算式:k 是 1 回傳 p,否則回傳 1-p
    return p if k == 1 else (1 - p)


def categorical_pmf(k, probs):
    """類別分布(多類別版的伯努利):第k類的機率,直接查表回傳probs[k]"""
    return probs[k]


def poisson_pmf(k, lam):
    """卜瓦松分布:在固定時間/空間內,事件發生k次的機率,lam是平均發生率"""
    # 公式 λ^k * e^(-λ) / k!;例如每小時平均來 1.5 位客人,剛好來 2 位的機率
    return (lam ** k) * math.exp(-lam) / factorial(k)


def uniform_pdf(x, a, b):
    """均勻分布的密度函數:在[a,b]區間內密度固定是1/(b-a),區間外是0"""
    # 注意這是「密度」不是機率:區間寬度 (b-a) 越小,密度越高,但密度乘上寬度加總永遠是 1
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def normal_pdf(x, mu, sigma):
    """常態(高斯)分布的密度函數,mu是平均值、sigma是標準差,決定鐘形曲線的中心跟寬度"""
    # 公式:1/(σ√(2π)) * exp(-(x-μ)²/(2σ²)),這裡拆成前面的係數 coeff 與指數 exponent 兩段
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2  # 離平均越遠,指數越負,密度越小
    # 標準常態(μ=0, σ=1)在 x=0 的密度 ≈ 0.3989;密度可以大於 1,因為它不是機率
    return coeff * math.exp(exponent)


# ---------- Step 4: 抽樣 ----------

def sample_bernoulli(p, n=1):
    """從Bernoulli(p)抽n個樣本:均勻隨機數落在[0,p)的機率剛好是p,藉此模擬成功/失敗"""
    # random.random() 均勻落在 [0,1),小於 p 的機率就是 p;`_` 表示迴圈變數不需要用
    return [1 if random.random() < p else 0 for _ in range(n)]


def sample_categorical(probs, n=1):
    """從類別分布抽n個樣本:把機率累加成累積區間,隨機數落在哪個區間就回傳對應類別"""
    # 例:probs=[0.5,0.3,0.2] → 累積 [0.5,0.8,1.0];隨機數 0.65 落在 (0.5,0.8],回傳類別 1
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        # enumerate 同時取得編號 i 與累積值 c;第一個「r <= c」的位置就是抽中的類別
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples


def sample_normal_box_muller(mu, sigma, n=1):
    """用Box-Muller轉換,把兩個均勻分布隨機數轉成常態分布樣本,
    再用 mu + sigma*z 平移縮放成目標分布"""
    samples = []
    for _ in range(n):
        u1 = random.random()  # 兩個獨立的均勻隨機數
        u2 = random.random()
        # Box-Muller 公式:z = sqrt(-2 ln u1) * cos(2π u2),得到標準常態(平均0、標準差1)的一個樣本
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        # 標準常態乘上 sigma 放大寬度、加上 mu 平移中心,得到目標常態分布的樣本
        samples.append(mu + sigma * z)
    return samples


# ---------- Step 6: 中央極限定理 demo ----------

def demonstrate_clt(dist_fn, n_samples, n_averages):
    """中央極限定理demo:不管dist_fn原本長什麼分布,只要每組平均n_samples個值,
    重複做n_averages組,這些「平均值」的分布會趨近常態分布(鐘形曲線)"""
    # dist_fn 是「每次呼叫回傳一個隨機數」的函數(函數當參數傳入)
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]  # 抽一組 n_samples 個
        averages.append(sum(samples) / len(samples))     # 這組的平均值
    # 關鍵在 n_samples(每組平均幾個)夠大,不是 n_averages(重複幾組);
    # 重複次數只是讓長條圖畫得更平滑
    return averages


# ---------- Demo ----------

def demo_basics():
    # 撲克牌:12 張人頭牌(J、Q、K 各 4 張)裡有 4 張是 K
    # P(K 且 人頭牌) = 4/52(K 一定是人頭牌),P(人頭牌) = 12/52,答案 = 1/3
    p = conditional_probability(4 / 52, 12 / 52)
    print(f"P(King | Face card) = {p:.4f}")


def demo_pmf_pdf():
    print("Bernoulli P(X=1|p=0.3) =", bernoulli_pmf(1, 0.3))          # 0.3
    print("Poisson P(X=2|lambda=1.5) =", poisson_pmf(2, 1.5))
    print("Normal pdf at x=0, mu=0, sigma=1 =", normal_pdf(0, 0, 1))  # ≈ 0.3989


def demo_expectation():
    die_values = [1, 2, 3, 4, 5, 6]
    die_probs = [1 / 6] * 6  # [1/6] * 6 是把同一個數字重複 6 次,共 6 個
    mu = expected_value(die_values, die_probs)
    var = variance(die_values, die_probs)
    # SD(標準差)= 變異數開根號,單位跟原本的數字一致,比較好理解
    print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var ** 0.5:.4f}")


def demo_softmax():
    logits = [2.0, 1.0, 0.1]
    probs = softmax(logits)
    log_probs = log_softmax(logits)
    # 巢狀 f-string:把每個機率轉成 4 位小數的字串再一起印出
    print("softmax:", [f"{p:.4f}" for p in probs])
    print("log_softmax:", [f"{p:.4f}" for p in log_probs])
    # 目標是第 0 類(分數最高、機率約 0.659),loss = -log(0.659) ≈ 0.417
    print("cross_entropy(target=0):", cross_entropy_loss(logits, 0))


def demo_clt():
    # lambda: random.randint(1, 6) 是一個「每次呼叫就擲一次骰子」的小函數
    averages = demonstrate_clt(lambda: random.randint(1, 6), n_samples=30, n_averages=1000)
    mean_of_averages = sum(averages) / len(averages)
    print(f"CLT demo: mean of 1000 averages-of-30-dice-rolls = {mean_of_averages:.4f} (should be near 3.5)")


if __name__ == "__main__":
    # 直接執行這個檔案時才會跑;沒有固定亂數種子,demo_clt 的結果每次略有不同
    demo_basics()
    demo_pmf_pdf()
    demo_expectation()
    demo_softmax()
    demo_clt()

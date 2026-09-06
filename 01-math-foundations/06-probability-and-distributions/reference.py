"""
Lesson 6: Probability and Distributions
機率與分布——涵蓋機率公理、PMF/PDF、期望值/變異數、抽樣、
softmax/log-softmax/cross-entropy、中央極限定理(CLT) demo

核心主軸:AI系統本質上都在跟不確定性打交道——分類模型輸出的是機率
(softmax),訓練用的loss是機率的期望值(cross-entropy),權重初始化
靠機率分布(normal),這支程式從機率論最底層的公理開始,一路連到
訓練神經網路每一步實際在做的事。
"""

import math
import random


# ---------- Step 1: 機率基礎 ----------

def factorial(n):
    """n! = n * (n-1) * ... * 1,用來算排列組合"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def combinations(n, k):
    """從n個東西裡選k個,不管順序,C(n,k) = n! / (k! * (n-k)!)"""
    return factorial(n) // (factorial(k) * factorial(n - k))


def conditional_probability(p_a_and_b, p_b):
    """P(A|B) = P(A且B) / P(B) —— 已知B發生的前提下,A發生的機率"""
    return p_a_and_b / p_b


# ---------- Step 2: PMF / PDF ----------

def bernoulli_pmf(k, p):
    """伯努利分布:k=1(成功)的機率是p,k=0(失敗)的機率是1-p"""
    return p if k == 1 else (1 - p)


def categorical_pmf(k, probs):
    """類別分布(多類別版的伯努利):第k類的機率,直接查表回傳probs[k]"""
    return probs[k]


def poisson_pmf(k, lam):
    """卜瓦松分布:在固定時間/空間內,事件發生k次的機率,lam是平均發生率"""
    return (lam ** k) * math.exp(-lam) / factorial(k)


def uniform_pdf(x, a, b):
    """均勻分布的密度函數:在[a,b]區間內密度固定是1/(b-a),區間外是0"""
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def normal_pdf(x, mu, sigma):
    """常態(高斯)分布的密度函數,mu是平均值、sigma是標準差,決定鐘形曲線的中心跟寬度"""
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)


# ---------- Step 3: 期望值 / 變異數 ----------

def expected_value(values, probabilities):
    """期望值 E[X]:每個結果乘上自己的機率再加總,機率加權平均"""
    return sum(v * p for v, p in zip(values, probabilities))


def variance(values, probabilities):
    """變異數 Var(X) = E[(X-mu)^2]:每個結果離平均值的差距平方,再取機率加權平均
    (跟展開版 E[X^2]-(E[X])^2 數學上等價,這裡用定義版,邏輯較直觀)"""
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))


# ---------- Step 4: 抽樣 ----------

def sample_bernoulli(p, n=1):
    """從Bernoulli(p)抽n個樣本:均勻隨機數落在[0,p)的機率剛好是p,藉此模擬成功/失敗"""
    return [1 if random.random() < p else 0 for _ in range(n)]


def sample_categorical(probs, n=1):
    """從類別分布抽n個樣本:把機率累加成累積區間,隨機數落在哪個區間就回傳對應類別"""
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
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
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples


# ---------- Step 5: softmax / log-softmax / cross-entropy ----------

def softmax(logits):
    """把原始分數(logits)轉成合法機率分布(全部在0~1之間,加總=1)。
    先減掉最大值再取exp,是數值穩定技巧:避免exp(大數字)直接溢位,
    因為分子分母同時除以同一個常數,比例完全不變"""
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(logits):
    """直接算log機率,不用先softmax再取log(避免機率太小時log(0)出現負無窮)。
    用log-sum-exp技巧把取指數跟取log合併簡化成一步"""
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]


def cross_entropy_loss(logits, target_index):
    """分類問題的loss:正確類別的log機率取負號。
    模型對正確答案越沒信心(機率越低),loss越大,值域是[0,+∞)"""
    log_probs = log_softmax(logits)
    return -log_probs[target_index]


# ---------- Step 6: 中央極限定理 demo ----------

def demonstrate_clt(dist_fn, n_samples, n_averages):
    """中央極限定理demo:不管dist_fn原本長什麼分布,只要每組平均n_samples個值,
    重複做n_averages組,這些「平均值」的分布會趨近常態分布(鐘形曲線)"""
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages


# ---------- Demo ----------

def demo_basics():
    p = conditional_probability(4 / 52, 12 / 52)
    print(f"P(King | Face card) = {p:.4f}")


def demo_pmf_pdf():
    print("Bernoulli P(X=1|p=0.3) =", bernoulli_pmf(1, 0.3))
    print("Poisson P(X=2|lambda=1.5) =", poisson_pmf(2, 1.5))
    print("Normal pdf at x=0, mu=0, sigma=1 =", normal_pdf(0, 0, 1))


def demo_expectation():
    die_values = [1, 2, 3, 4, 5, 6]
    die_probs = [1 / 6] * 6
    mu = expected_value(die_values, die_probs)
    var = variance(die_values, die_probs)
    print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var ** 0.5:.4f}")


def demo_softmax():
    logits = [2.0, 1.0, 0.1]
    probs = softmax(logits)
    log_probs = log_softmax(logits)
    print("softmax:", [f"{p:.4f}" for p in probs])
    print("log_softmax:", [f"{p:.4f}" for p in log_probs])
    print("cross_entropy(target=0):", cross_entropy_loss(logits, 0))


def demo_clt():
    averages = demonstrate_clt(lambda: random.randint(1, 6), n_samples=30, n_averages=1000)
    mean_of_averages = sum(averages) / len(averages)
    print(f"CLT demo: mean of 1000 averages-of-30-dice-rolls = {mean_of_averages:.4f} (should be near 3.5)")


if __name__ == "__main__":
    demo_basics()
    demo_pmf_pdf()
    demo_expectation()
    demo_softmax()
    demo_clt()

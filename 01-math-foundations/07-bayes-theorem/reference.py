"""
Lesson 7: Bayes' Theorem
貝氏定理——從條件機率推出貝氏定理,實作Naive Bayes文字分類器(含Laplace
平滑跟log機率),比較MLE跟MAP估計,示範Beta-Binomial共軛先驗的序列更新。

核心主軸:機率不只是「預測」,貝氏定理教的是「看到新證據後怎麼更新信念」。
垃圾郵件過濾器、醫療診斷、任何會輸出「信心分數」的機器學習模型,背後都是
同一套推理邏輯:先驗(prior) + 證據(likelihood) -> 後驗(posterior)。
"""

import math
from collections import defaultdict


# === 核心(手刻層,逐行講解+手打練熟) ===
# ---------- Step 1: 貝氏定理本體 ----------

def bayes(prior, likelihood, false_positive_rate):
    """P(A|B) = P(B|A)*P(A) / P(B)
    prior: P(A),看到證據前的信念(例如:得病的機率)
    likelihood: P(B|A),假設成立時看到證據的機率(例如:得病時篩檢陽性的機率)
    false_positive_rate: P(B|not A),假設不成立時仍看到證據的機率(健康卻篩檢陽性)
    用全機率公式展開分母:P(B) = P(B|A)*P(A) + P(B|not A)*P(not A)
    """
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior


# === 理解層(講邏輯+demo驗證,不逐行摳) ===
# ---------- Step 2: Naive Bayes 文字分類器 ----------

class NaiveBayes:
    """樸素貝氏分類器:假設每個詞出現的機率,在已知類別的條件下互相獨立
    (這個假設其實不成立——例如"New"跟"York"明顯相關——但實務上效果依然很好,
    因為分類只需要「排序哪個類別分數最高」,不需要機率本身校準得很準)"""

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing  # Laplace平滑係數,避免沒看過的詞讓機率變成0
        self.class_counts = defaultdict(int)          # 每個類別看過幾篇文件
        self.word_counts = defaultdict(lambda: defaultdict(int))  # 每個類別裡每個詞出現次數
        self.class_word_totals = defaultdict(int)      # 每個類別的總詞數
        self.vocab = set()                              # 全部看過的詞彙表

    def train(self, documents, labels):
        """數數字而已:每篇文件屬於哪個類別、裡面每個詞出現幾次,全部累加起來"""
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def predict(self, document):
        """對每個類別算一個分數:log(P(類別)) + 每個詞的log(P(詞|類別)) 加總,
        分數最高的類別就是預測結果。用log機率是因為好幾十個機率連乘會下溢
        (跟Lesson 6的log-softmax同一個原理),連加不會有這個問題,而且排序結果不變。"""
        words = document.lower().split()
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)
        best_class = None
        best_score = float("-inf")
        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total_docs)  # log(prior)
            for word in words:
                count = self.word_counts[cls].get(word, 0)
                total = self.class_word_totals[cls]
                # Laplace平滑:分子分母都加一點,確保沒看過的詞不會讓機率變0
                score += math.log((count + self.smoothing) / (total + self.smoothing * vocab_size))
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class


# ---------- Step 3: MLE vs MAP ----------

def mle_estimate(successes, trials):
    """最大似然估計:直接用觀察到的比例,沒有任何先驗信念"""
    return successes / trials


def map_estimate(successes, trials, prior_alpha, prior_beta):
    """最大後驗估計:用Beta(alpha, beta)當先驗,後驗均值 = (alpha+successes) / (alpha+beta+trials)
    先驗係數越大,代表你越相信參數應該在prior_alpha/(prior_alpha+prior_beta)附近,
    資料量越小時,先驗的影響力越大——這就是為什麼MAP等於加了正則化的MLE。"""
    return (prior_alpha + successes) / (prior_alpha + prior_beta + trials)


# ---------- Step 4: Beta-Binomial 共軛先驗、序列更新 ----------

def beta_update(alpha, beta, successes, failures):
    """Beta分布是Bernoulli/Binomial的共軛先驗:後驗還是Beta分布,
    更新規則簡單到不可思議——直接把成功/失敗次數加到alpha/beta上,不用做任何積分"""
    return alpha + successes, beta + failures


def beta_mean(alpha, beta):
    """Beta(alpha, beta)的期望值,alpha+beta越大代表分布越集中(對這個機率值越有信心)"""
    return alpha / (alpha + beta)


# ---------- Demo ----------

def demo_bayes_medical():
    result = bayes(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
    print(f"P(sick|positive) = {result:.4f} (罕見疾病,即使測試準確率99%,陽性後仍只有約1%真的得病)")


def demo_naive_bayes():
    train_docs = [
        "win free money now",
        "free lottery ticket winner",
        "claim your prize today free",
        "urgent offer free cash",
        "congratulations you won free",
        "meeting tomorrow at noon",
        "project update attached",
        "can we schedule a call",
        "quarterly report review",
        "lunch on thursday sounds good",
        "team standup notes attached",
        "please review the pull request",
    ]
    train_labels = [
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham", "ham", "ham",
    ]
    classifier = NaiveBayes()
    classifier.train(train_docs, train_labels)

    test_messages = [
        "free money waiting for you",
        "meeting rescheduled to friday",
        "you won a free prize",
        "please review the attached report",
    ]
    print("Naive Bayes spam分類結果:")
    for msg in test_messages:
        print(f"  '{msg}' -> {classifier.predict(msg)}")


def demo_mle_vs_map():
    # 情境:丟10次硬幣,7次正面。MLE直接說偏向正面(0.7);
    # MAP因為先驗Beta(2,2)相信硬幣接近公平,結果會被拉回0.5一點
    mle = mle_estimate(successes=7, trials=10)
    map_result = map_estimate(successes=7, trials=10, prior_alpha=2, prior_beta=2)
    print(f"MLE估計(10次丟硬幣7次正面) = {mle:.4f}")
    print(f"MAP估計(先驗Beta(2,2),相信硬幣接近公平) = {map_result:.4f}")


def demo_sequential_update():
    # Day1: 完全沒資訊,均勻先驗Beta(1,1)
    alpha, beta = 1, 1
    print(f"Day 1 先驗: Beta({alpha},{beta}), 均值={beta_mean(alpha, beta):.4f}")

    # Day2: 觀察到7正3反
    alpha, beta = beta_update(alpha, beta, successes=7, failures=3)
    print(f"Day 2 後驗: Beta({alpha},{beta}), 均值={beta_mean(alpha, beta):.4f}")

    # Day3: 昨天的後驗變成今天的先驗,再觀察5正5反
    alpha, beta = beta_update(alpha, beta, successes=5, failures=5)
    print(f"Day 3 後驗: Beta({alpha},{beta}), 均值={beta_mean(alpha, beta):.4f}")


if __name__ == "__main__":
    demo_bayes_medical()
    print()
    demo_naive_bayes()
    print()
    demo_mle_vs_map()
    print()
    demo_sequential_update()

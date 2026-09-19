"""
Lesson 7: Bayes' Theorem
貝氏定理——從條件機率推出貝氏定理,實作Naive Bayes文字分類器(含Laplace
平滑跟log機率),比較MLE跟MAP估計,示範Beta-Binomial共軛先驗的序列更新。

核心主軸:機率不只是「預測」,貝氏定理教的是「看到新證據後怎麼更新信念」。
垃圾郵件過濾器、醫療診斷、任何會輸出「信心分數」的機器學習模型,背後都是
同一套推理邏輯:先驗(prior) + 證據(likelihood) -> 後驗(posterior)。

名詞速記:
    先驗 prior          : 看到證據「之前」對某件事的信念(例如人群中得病的比例)
    似然 likelihood     : 如果假設成立,看到這個證據的機率
    證據 evidence       : 不管假設成不成立,看到這個證據的總機率(當作分母,讓後驗加總為 1)
    後驗 posterior      : 看到證據「之後」更新過的信念,也就是最終要算的答案
    公式 posterior = likelihood × prior / evidence
"""

import math                            # log 用來做「機率連乘 → log 連加」
from collections import defaultdict    # 找不到 key 時自動建立預設值的字典,計數時很方便


# === 🔴 ===
# ---------- Step 1: 貝氏定理本體 ----------

def bayes(prior, likelihood, false_positive_rate):
    """P(A|B) = P(B|A)*P(A) / P(B)
    prior: P(A),看到證據前的信念(例如:得病的機率)
    likelihood: P(B|A),假設成立時看到證據的機率(例如:得病時篩檢陽性的機率)
    false_positive_rate: P(B|not A),假設不成立時仍看到證據的機率(健康卻篩檢陽性)
    用全機率公式展開分母:P(B) = P(B|A)*P(A) + P(B|not A)*P(not A)
    """
    # 分母 evidence:「陽性」有兩種來源——真的得病而陽性、沒病卻誤判陽性,兩者相加
    # (1 - prior) 是「沒得病」的機率
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    # 分子只算「真的得病而且陽性」這一種來源,除以所有陽性的來源,就是「陽性時真的得病」的機率
    # 例:prior=0.0001、likelihood=0.99、false_positive=0.01 → 分子 0.000099、分母 0.010098,後驗約 0.0098
    posterior = likelihood * prior / evidence
    return posterior


# === 🟡 ===
# ---------- Step 2: Naive Bayes 文字分類器 ----------

class NaiveBayes:
    """樸素貝氏分類器:假設每個詞出現的機率,在已知類別的條件下互相獨立
    (這個假設其實不成立——例如"New"跟"York"明顯相關——但實務上效果依然很好,
    因為分類只需要「排序哪個類別分數最高」,不需要機率本身校準得很準)"""

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing  # Laplace平滑係數,避免沒看過的詞讓機率變成0
        # defaultdict(int):讀取不存在的 key 時自動當成 0,可以直接 += 1 計數
        self.class_counts = defaultdict(int)          # 每個類別看過幾篇文件
        # 兩層字典:word_counts[類別][詞] = 次數;內層預設值也是 defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))  # 每個類別裡每個詞出現次數
        self.class_word_totals = defaultdict(int)      # 每個類別的總詞數
        self.vocab = set()                              # 全部看過的詞彙表

    def train(self, documents, labels):
        """數數字而已:每篇文件屬於哪個類別、裡面每個詞出現幾次,全部累加起來"""
        # 訓練 = 純粹計數,沒有梯度、沒有迭代,一次掃過資料就完成
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            # .lower() 轉小寫、.split() 依空白切成單字 list;"Free money" → ["free", "money"]
            words = doc.lower().split()
            # 外層迴圈是文件、內層迴圈是文件裡的每個詞;三個計數器都在這裡一起累加
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)  # set 會自動去除重複

    def predict(self, document):
        """對每個類別算一個分數:log(P(類別)) + 每個詞的log(P(詞|類別)) 加總,
        分數最高的類別就是預測結果。用log機率是因為好幾十個機率連乘會下溢
        (跟Lesson 6的log-softmax同一個原理),連加不會有這個問題,而且排序結果不變。"""
        words = document.lower().split()
        total_docs = sum(self.class_counts.values())  # 訓練文件總數,用來算先驗
        vocab_size = len(self.vocab)                   # 詞彙表大小,平滑的分母要用到
        best_class = None
        best_score = float("-inf")  # 負無窮,任何真實分數都比它大,確保第一個類別一定會被記下來
        # 對每個候選類別(spam、ham)各算一個分數,最後挑最高的
        for cls in self.class_counts:
            # 先驗:這個類別佔全部文件的比例,取 log
            score = math.log(self.class_counts[cls] / total_docs)  # log(prior)
            for word in words:
                # .get(word, 0):沒看過的詞回傳 0,不會報錯
                count = self.word_counts[cls].get(word, 0)
                total = self.class_word_totals[cls]
                # Laplace平滑:分子分母都加一點,確保沒看過的詞不會讓機率變0
                # 分子 +smoothing(每個詞都當成至少出現過一點),分母 +smoothing*詞彙表大小(讓機率仍加總為 1)
                score += math.log((count + self.smoothing) / (total + self.smoothing * vocab_size))
            # 分數最高的類別勝出
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class


# ---------- Step 3: MLE vs MAP ----------

def mle_estimate(successes, trials):
    """最大似然估計:直接用觀察到的比例,沒有任何先驗信念"""
    # 例:丟 10 次、7 次正面 → 0.7
    # 缺點:資料很少時容易極端,例如丟 2 次都正面會估出 100% 正面
    return successes / trials


def map_estimate(successes, trials, prior_alpha, prior_beta):
    """最大後驗估計:用Beta(alpha, beta)當先驗,後驗均值 = (alpha+successes) / (alpha+beta+trials)
    先驗係數越大,代表越相信參數應該在prior_alpha/(prior_alpha+prior_beta)附近,
    資料量越小時,先驗的影響力越大——這就是為什麼MAP等於加了正則化的MLE。"""
    # 可以想成:先「假裝」已經看過 alpha 次成功、beta 次失敗,再加上真實資料
    # Beta(2,2):假裝已看過 2 正 2 反,7 正 3 反 → (2+7)/(2+2+10) = 9/14 ≈ 0.643,比 MLE 的 0.7 更靠近 0.5
    return (prior_alpha + successes) / (prior_alpha + prior_beta + trials)


# ---------- Step 4: Beta-Binomial 共軛先驗、序列更新 ----------

def beta_update(alpha, beta, successes, failures):
    """Beta分布是Bernoulli/Binomial的共軛先驗:後驗還是Beta分布,
    更新規則簡單到不可思議——直接把成功/失敗次數加到alpha/beta上,不用做任何積分"""
    # 「共軛」的意思:先驗跟後驗是同一個分布家族,所以可以一天一天連續更新
    # 回傳 tuple (新 alpha, 新 beta),呼叫端用兩個變數同時接
    return alpha + successes, beta + failures


def beta_mean(alpha, beta):
    """Beta(alpha, beta)的期望值,alpha+beta越大代表分布越集中(對這個機率值越有信心)"""
    # 例:Beta(8,4) 的均值 = 8/12 ≈ 0.667,代表目前估計硬幣正面機率約 66.7%
    return alpha / (alpha + beta)


# ---------- Demo ----------

def demo_bayes_medical():
    # 經典反直覺例子:疾病盛行率 0.01%、檢測對病人 99% 陽性、對健康人 1% 誤判陽性
    # 每 10000 人只有 1 人得病;健康的 9999 人裡約 100 人誤判陽性,遠多於真正得病的那 1 人
    result = bayes(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
    print(f"P(sick|positive) = {result:.4f} (罕見疾病,即使測試準確率99%,陽性後仍只有約1%真的得病)")


def demo_naive_bayes():
    # 12 封訓練郵件:前 5 封是垃圾(spam)、後 7 封是正常(ham)
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
    # 先建立分類器,再用 12 封標好類別的郵件訓練(只是數數字)
    classifier = NaiveBayes()
    classifier.train(train_docs, train_labels)

    # 四封測試郵件,前兩封的詞跟垃圾/正常郵件明顯相關,後面兩封也各有偏向
    test_messages = [
        "free money waiting for you",
        "meeting rescheduled to friday",
        "you won a free prize",
        "please review the attached report",
    ]
    # 預期前兩封看起來像垃圾/正常各自分到對應類別,可以跟 sklearn 版的結果對照
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
    print(f"Day 1 先驗: Beta({alpha},{beta}), 均值={beta_mean(alpha, beta):.4f}")  # 0.5

    # Day2: 觀察到7正3反
    alpha, beta = beta_update(alpha, beta, successes=7, failures=3)  # Beta(8,4),均值 8/12
    print(f"Day 2 後驗: Beta({alpha},{beta}), 均值={beta_mean(alpha, beta):.4f}")

    # Day3: 昨天的後驗變成今天的先驗,再觀察5正5反
    alpha, beta = beta_update(alpha, beta, successes=5, failures=5)  # Beta(13,9),均值 13/22
    print(f"Day 3 後驗: Beta({alpha},{beta}), 均值={beta_mean(alpha, beta):.4f}")


if __name__ == "__main__":
    # 直接執行這個檔案時才會跑;依序示範四個主題,print() 空行分隔輸出
    demo_bayes_medical()
    print()
    demo_naive_bayes()
    print()
    demo_mle_vs_map()
    print()
    demo_sequential_update()

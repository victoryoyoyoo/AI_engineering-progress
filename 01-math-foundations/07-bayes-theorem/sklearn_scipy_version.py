"""
Lesson 7: 用 scikit-learn / SciPy 版本對照
reference.py 手刻的 NaiveBayes class、beta_update / beta_mean,
在 scikit-learn 與 SciPy 裡各是幾行呼叫。
"""
from sklearn.feature_extraction.text import CountVectorizer  # 文字轉詞頻向量
from sklearn.naive_bayes import MultinomialNB                # 適合詞頻資料的樸素貝氏分類器
from scipy.stats import beta as scipy_beta                   # Beta 分布物件,改名避免與變數名 beta 撞名


def demo_sklearn_naive_bayes():
    # 訓練資料與測試資料跟 reference.py 完全相同,方便對照兩邊的分類結果是否一致
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
    test_messages = [
        "free money waiting for you",
        "meeting rescheduled to friday",
        "you won a free prize",
        "please review the attached report",
    ]

    # CountVectorizer 負責把句子轉成詞頻向量,對應手刻的 train()/predict() 裡數詞頻的部分
    vectorizer = CountVectorizer()
    # fit_transform:先掃過訓練資料建立詞彙表(fit),再把每封信轉成「每個詞出現幾次」的向量(transform)
    # 結果是稀疏矩陣,shape (12 封信, 詞彙表大小)
    X_train = vectorizer.fit_transform(train_docs)

    # MultinomialNB 內部自動處理 Laplace smoothing + log機率,對應手刻的整個 NaiveBayes class
    clf = MultinomialNB()
    clf.fit(X_train, train_labels)  # fit = 訓練;在這裡就是數數字,跟 reference.py 的 train() 同一件事

    # 測試資料只能用 transform(不能再 fit),沿用訓練時建立的詞彙表,沒看過的詞會被忽略
    X_test = vectorizer.transform(test_messages)
    predictions = clf.predict(X_test)

    print("scikit-learn Naive Bayes 分類結果:")
    for msg, pred in zip(test_messages, predictions):
        print(f"  '{msg}' -> {pred}")


def demo_beta_distribution():
    # scipy.stats.beta 對應手刻的 beta_update()/beta_mean(),
    # .mean() 直接算期望值,不用自己寫 a/(a+b)
    # a、b 就是 Beta 分布的 alpha、beta;Beta(1,1) 是均勻先驗
    prior = scipy_beta(a=1, b=1)
    # 更新規則是把成功/失敗次數加到 a、b 上:Beta(1,1) 看到 7 正 3 反 → Beta(8,4)
    posterior_day2 = scipy_beta(a=8, b=4)
    # 再看到 5 正 5 反 → Beta(13,9)
    posterior_day3 = scipy_beta(a=13, b=9)

    print(f"SciPy Beta(1,1) 均值  = {prior.mean():.4f}")   # 0.5
    print(f"SciPy Beta(8,4) 均值  = {posterior_day2.mean():.4f}")   # 8/12 ≈ 0.6667
    print(f"SciPy Beta(13,9) 均值 = {posterior_day3.mean():.4f}")  # 13/22 ≈ 0.5909


if __name__ == "__main__":
    demo_sklearn_naive_bayes()
    print()
    demo_beta_distribution()

"""
Lesson 7: 用 scikit-learn / SciPy 版本對照
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from scipy.stats import beta as scipy_beta


def demo_sklearn_naive_bayes():
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
    X_train = vectorizer.fit_transform(train_docs)

    # MultinomialNB 內部自動處理 Laplace smoothing + log機率,對應手刻的整個 NaiveBayes class
    clf = MultinomialNB()
    clf.fit(X_train, train_labels)

    X_test = vectorizer.transform(test_messages)
    predictions = clf.predict(X_test)

    print("scikit-learn Naive Bayes 分類結果:")
    for msg, pred in zip(test_messages, predictions):
        print(f"  '{msg}' -> {pred}")


def demo_beta_distribution():
    # scipy.stats.beta 對應手刻的 beta_update()/beta_mean(),
    # .mean() 直接算期望值,不用自己寫 a/(a+b)
    prior = scipy_beta(a=1, b=1)
    posterior_day2 = scipy_beta(a=8, b=4)
    posterior_day3 = scipy_beta(a=13, b=9)

    print(f"SciPy Beta(1,1) 均值  = {prior.mean():.4f}")
    print(f"SciPy Beta(8,4) 均值  = {posterior_day2.mean():.4f}")
    print(f"SciPy Beta(13,9) 均值 = {posterior_day3.mean():.4f}")


if __name__ == "__main__":
    demo_sklearn_naive_bayes()
    print()
    demo_beta_distribution()

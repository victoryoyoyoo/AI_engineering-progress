"""
Lesson 7 練習:貝氏定理(Bayes' Theorem)
"""


def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior


if __name__ == "__main__":
    # 醫療檢測範例:1萬人中1人得病,測試準確率99%
    result = bayes(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
    print(f"P(sick|positive) = {result:.4f}")

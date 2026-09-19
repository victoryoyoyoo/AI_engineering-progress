"""
Lesson 6:用 NumPy / SciPy 版本對照
reference.py 手刻的期望值、變異數、常態分布密度、softmax、log-softmax,
在 NumPy / SciPy 裡都有現成函式,這份檔案逐一對照,結果應該與手刻版一致。
"""
import numpy as np                                       # 陣列與向量運算
from scipy import stats                                  # 各種機率分布(pdf、cdf、抽樣)
from scipy.special import softmax as scipy_softmax       # 改名 import,避免跟其他 softmax 撞名
from scipy.special import log_softmax as scipy_log_softmax


def demo_expected_value_variance():
    die_values = np.array([1, 2, 3, 4, 5, 6])
    die_probs = np.array([1 / 6] * 6)  # 公平骰子,每面機率 1/6

    # 期望值:機率加權平均,對應手刻的 expected_value()
    # np.average(a, weights=w) = sum(a*w)/sum(w);權重加總是 1 時就是機率加權平均,結果 3.5
    mu = np.average(die_values, weights=die_probs)

    # 變異數:用 E[(X-mu)^2]，對應手刻的 variance()
    # die_values - mu 是陣列減純量(broadcasting),每個元素各自減掉 mu 再平方,結果 35/12 ≈ 2.9167
    var = np.average((die_values - mu) ** 2, weights=die_probs)

    print(f"NumPy: E[X]={mu:.4f}, Var(X)={var:.4f}")


def demo_normal_pdf():
    # scipy.stats.norm.pdf 對應手刻的 normal_pdf()
    # loc 是平均值 mu,scale 是標準差 sigma;標準常態在 x=0 的密度 ≈ 0.3989
    pdf_value = stats.norm.pdf(0, loc=0, scale=1)
    print(f"SciPy normal pdf at x=0, mu=0, sigma=1 = {pdf_value:.4f}")


def demo_softmax():
    logits = np.array([2.0, 1.0, 0.1])

    # scipy.special.softmax 對應手刻的 softmax();內部同樣做了減最大值的數值穩定處理
    probs = scipy_softmax(logits)

    # scipy.special.log_softmax 對應手刻的 log_softmax();直接算 log 機率,不會有 log(0) 問題
    log_probs = scipy_log_softmax(logits)

    # np.round(x, 4) 對整個陣列四捨五入到 4 位小數
    print("SciPy softmax:", np.round(probs, 4))
    print("SciPy log_softmax:", np.round(log_probs, 4))


if __name__ == "__main__":
    demo_expected_value_variance()
    demo_normal_pdf()
    demo_softmax()

"""
Lesson 6:用 NumPy / SciPy 版本對照
"""
import numpy as np
from scipy import stats
from scipy.special import softmax as scipy_softmax
from scipy.special import log_softmax as scipy_log_softmax


def demo_expected_value_variance():
    die_values = np.array([1, 2, 3, 4, 5, 6])
    die_probs = np.array([1 / 6] * 6)

    # 期望值:機率加權平均,對應手刻的 expected_value()
    mu = np.average(die_values, weights=die_probs)

    # 變異數:用 E[(X-mu)^2]，對應手刻的 variance()
    var = np.average((die_values - mu) ** 2, weights=die_probs)

    print(f"NumPy: E[X]={mu:.4f}, Var(X)={var:.4f}")


def demo_normal_pdf():
    # scipy.stats.norm.pdf 對應手刻的 normal_pdf()
    pdf_value = stats.norm.pdf(0, loc=0, scale=1)
    print(f"SciPy normal pdf at x=0, mu=0, sigma=1 = {pdf_value:.4f}")


def demo_softmax():
    logits = np.array([2.0, 1.0, 0.1])

    # scipy.special.softmax 對應手刻的 softmax()
    probs = scipy_softmax(logits)

    # scipy.special.log_softmax 對應手刻的 log_softmax()
    log_probs = scipy_log_softmax(logits)

    print("SciPy softmax:", np.round(probs, 4))
    print("SciPy log_softmax:", np.round(log_probs, 4))


if __name__ == "__main__":
    demo_expected_value_variance()
    demo_normal_pdf()
    demo_softmax()

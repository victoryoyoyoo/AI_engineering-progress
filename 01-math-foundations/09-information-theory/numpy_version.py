"""
Lesson 9: 用NumPy對照
"""
import numpy as np


def np_entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0
    result = np.zeros_like(p)
    result[mask] = p[mask] * np.log(p[mask])
    return -result.sum()


def np_cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return -(p[mask] * np.log(q[mask])).sum()


def np_kl_divergence(p, q):
    return np_cross_entropy(p, q) - np_entropy(p)


if __name__ == "__main__":
    true = np.array([0.7, 0.2, 0.1])
    pred = np.array([0.6, 0.25, 0.15])
    print(f"Entropy:    {np_entropy(true):.4f} nats")
    print(f"Cross-ent:  {np_cross_entropy(true, pred):.4f} nats")
    print(f"KL div:     {np_kl_divergence(true, pred):.4f} nats")

    # PyTorch的torch.nn.CrossEntropyLoss()內部做的事,就是這條np_cross_entropy
    # 的簡化版(P是one-hot時),外加softmax + log結合成一步(數值上更穩定)

import math


def information_content(p, base=2):
    if p <= 0:
        return float('inf')
    if p >= 1:
        return 0.0
    return -math.log(p) / math.log(base)


def entropy(probs, base=2):
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )


def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * (-math.log(qi) / math.log(base))
    return total


def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)


def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0 and margin_x[i] > 0 and margin_y[j] > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi


def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])


def perplexity(avg_cross_entropy, base="e"):
    if base == "e":
        return math.exp(avg_cross_entropy)
    return 2 ** avg_cross_entropy


if __name__ == "__main__":
    true_dist = [0.7, 0.2, 0.1]
    good_model = [0.6, 0.25, 0.15]

    print(f"H(true):          {entropy(true_dist):.4f} bits")
    print(f"CE(true, good):   {cross_entropy(true_dist, good_model):.4f} bits")
    print(f"KL(true || good): {kl_divergence(true_dist, good_model):.4f} bits")

    logits = [2.0, 1.0, 0.1]
    loss = cross_entropy_loss(true_class=0, logits=logits)
    print(f"CE loss: {loss:.4f} nats, Perplexity: {perplexity(loss):.2f}")

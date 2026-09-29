import math


def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))


def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))


print(softmax_naive([2.0, 1.0, 0.1]))
print(softmax_stable([2.0, 1.0, 0.1]))
print(softmax_stable([1000.0, 1001.0, 1002.0]))
print(logsumexp_naive([1.0, 2.0, 3.0]))
print(logsumexp_stable([1.0, 2.0, 3.0]))
print(logsumexp_stable([500.0, 501.0, 502.0]))

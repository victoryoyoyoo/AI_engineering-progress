import math


def l1_norm(x):
    return sum(abs(xi) for xi in x)


def l2_norm(x):
    return math.sqrt(sum(xi ** 2 for xi in x))


def lp_norm(x, p):
    if p == float('inf'):
        return max(abs(xi) for xi in x)
    return sum(abs(xi) ** p for xi in x) ** (1 / p)


def linf_norm(x):
    return max(abs(xi) for xi in x)


def l1_distance(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


def l2_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def linf_distance(a, b):
    return max(abs(ai - bi) for ai, bi in zip(a, b))


def dot_product(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def cosine_similarity(a, b):
    dot = dot_product(a, b)
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_distance(a, b):
    return 1.0 - cosine_similarity(a, b)


def normalize_vector(v):
    norm = l2_norm(v)
    if norm == 0:
        return v[:]
    return [vi / norm for vi in v]


def jaccard_similarity(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def kl_divergence(p, q):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * math.log(pi / qi)
    return total


def wasserstein_1d(p, q):
    n = len(p)
    cdf_p = [0.0] * n
    cdf_q = [0.0] * n
    cdf_p[0] = p[0]
    cdf_q[0] = q[0]
    for i in range(1, n):
        cdf_p[i] = cdf_p[i - 1] + p[i]
        cdf_q[i] = cdf_q[i - 1] + q[i]
    return sum(abs(cdf_p[i] - cdf_q[i]) for i in range(n))


a, b = [1, 1], [4, 5]
print(l1_distance(a, b), l2_distance(a, b), linf_distance(a, b))
print(l1_norm([6, 8]), l2_norm([6, 8]), linf_norm([6, 8]))
print(cosine_similarity([1, 2], [2, 4]), cosine_similarity([1, 2], [3, 1]))
print(normalize_vector([3, 4]))
print(jaccard_similarity({"cat", "dog", "fish"}, {"cat", "bird", "fish", "snake"}))
print(edit_distance("kitten", "sitting"), edit_distance("sun", "sat"))
print(kl_divergence([0.9, 0.1], [0.5, 0.5]), kl_divergence([0.5, 0.5], [0.9, 0.1]))
print(wasserstein_1d([0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]))

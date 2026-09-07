def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior


if __name__ == "__main__":
    result = bayes(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
    print(f"P(sick|positive) = {result:.4f}")

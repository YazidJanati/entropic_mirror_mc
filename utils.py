import matplotlib.pyplot as plt


def estimate_kl(logpdf, proposal, target_samples):
    return (logpdf(target_samples) - proposal.log_prob(target_samples)).mean()

def display_samples(samples, color='black', size=1., alpha=1.):
    plt.figure(figsize=(10, 5))
    plt.scatter(samples[:, 0], samples[:, 1], c=color, s=size, alpha=alpha)
    plt.show()

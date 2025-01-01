import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define the target distribution: a 2D Gaussian mixture
def target_distribution(x):
    mean1, cov1 = np.array([2, 2]), np.array([[0.5, 0.1], [0.1, 0.5]])
    mean2, cov2 = np.array([-2, -2]), np.array([[0.5, -0.1], [-0.1, 0.5]])
    pdf1 = multivariate_normal.pdf(x, mean=mean1, cov=cov1)
    pdf2 = multivariate_normal.pdf(x, mean=mean2, cov=cov2)
    return 0.5 * pdf1 + 0.5 * pdf2

# Gradient of the negative log target distribution
def grad_neg_log_p(x):
    mean1, cov1 = np.array([2, 2]), np.array([[0.5, 0.1], [0.1, 0.5]])
    mean2, cov2 = np.array([-2, -2]), np.array([[0.5, -0.1], [-0.1, 0.5]])
    
    pdf1 = multivariate_normal.pdf(x, mean=mean1, cov=cov1)
    pdf2 = multivariate_normal.pdf(x, mean=mean2, cov=cov2)
    
    grad1 = -np.linalg.inv(cov1) @ (x - mean1)
    grad2 = -np.linalg.inv(cov2) @ (x - mean2)
    
    grad = 0.5 * (grad1 * pdf1 + grad2 * pdf2) / (pdf1 + pdf2)
    return -grad  # Gradient of the negative log probability

# Hamiltonian Monte Carlo (HMC) Step
def hmc_step(current_position, grad_neg_log_p, step_size=0.1, num_steps=10):
    # Sample initial momentum from standard normal
    current_momentum = np.random.randn(*current_position.shape)
    proposed_position = np.copy(current_position)
    proposed_momentum = np.copy(current_momentum)
    
    # Leapfrog integration
    proposed_momentum -= 0.5 * step_size * grad_neg_log_p(proposed_position)
    for _ in range(num_steps):
        proposed_position += step_size * proposed_momentum
        if _ != num_steps - 1:  # Full update except for the last step
            proposed_momentum -= step_size * grad_neg_log_p(proposed_position)
    proposed_momentum -= 0.5 * step_size * grad_neg_log_p(proposed_position)
    proposed_momentum = -proposed_momentum  # Reverse momentum for Metropolis-Hastings

    # Compute Hamiltonian for current and proposed states
    current_U = -np.log(target_distribution(current_position) + 1e-9)  # Add small value for stability
    proposed_U = -np.log(target_distribution(proposed_position) + 1e-9)
    current_K = np.sum(current_momentum**2) / 2
    proposed_K = np.sum(proposed_momentum**2) / 2
    
    # Metropolis-Hastings acceptance step
    acceptance_prob = np.exp(current_U - proposed_U + current_K - proposed_K)
    if np.random.rand() < acceptance_prob:
        return proposed_position
    else:
        return current_position

# Visualization function
def plot_particles(samples, target_distribution, iteration):
    plt.figure(figsize=(8, 6))
    # Plot the target distribution
    x, y = np.mgrid[-5:5:.01, -5:5:.01]
    pos = np.dstack((x, y))
    plt.contourf(x, y, target_distribution(pos), levels=30, cmap='viridis', alpha=0.6)
    
    # Plot the samples
    plt.scatter(samples[:, 0], samples[:, 1], color='red', s=10)
    plt.title(f'HMC Samples at Iteration {iteration}')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Initialize samples from a Gaussian distribution
np.random.seed(42)
num_samples = 100
samples = np.random.randn(num_samples, 2)

# Perform HMC sampling and plot
num_iterations = 500
step_size = 0.1
num_steps = 20
hmc_samples = []

current_position = samples[0]

for i in range(num_iterations):
    current_position = hmc_step(current_position, grad_neg_log_p, step_size, num_steps)
    hmc_samples.append(current_position)
    if i % 100 == 0 or i == num_iterations - 1:
        plot_particles(np.array(hmc_samples), target_distribution, i + 1)

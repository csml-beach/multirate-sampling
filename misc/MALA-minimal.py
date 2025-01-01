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

# Gradient of the log target distribution
def grad_log_p(x):
    mean1, cov1 = np.array([2, 2]), np.array([[0.5, 0.1], [0.1, 0.5]])
    mean2, cov2 = np.array([-2, -2]), np.array([[0.5, -0.1], [-0.1, 0.5]])
    
    pdf1 = multivariate_normal.pdf(x, mean=mean1, cov=cov1)
    pdf2 = multivariate_normal.pdf(x, mean=mean2, cov=cov2)
    
    grad1 = -np.linalg.inv(cov1) @ (x - mean1)
    grad2 = -np.linalg.inv(cov2) @ (x - mean2)
    
    grad = 0.5 * (grad1 * pdf1 + grad2 * pdf2) / (pdf1 + pdf2)
    return grad

# Metropolis-Adjusted Langevin Algorithm (MALA) step
def mala_step(current_position, grad_log_p, step_size=0.1):
    # Compute the proposal
    proposal_mean = current_position + (step_size**2 / 2) * grad_log_p(current_position)
    proposal = proposal_mean + step_size * np.random.randn(*current_position.shape)
    
    # Metropolis-Hastings acceptance step
    current_log_prob = np.log(target_distribution(current_position) + 1e-9)
    proposal_log_prob = np.log(target_distribution(proposal) + 1e-9)
    
    # Transition probabilities
    proposal_to_current = -np.sum((current_position - (proposal + (step_size**2 / 2) * grad_log_p(proposal)))**2) / (2 * step_size**2)
    current_to_proposal = -np.sum((proposal - (current_position + (step_size**2 / 2) * grad_log_p(current_position)))**2) / (2 * step_size**2)
    
    # Compute acceptance probability
    acceptance_prob = np.exp(proposal_log_prob - current_log_prob + proposal_to_current - current_to_proposal)
    
    if np.random.rand() < acceptance_prob:
        return proposal
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
    plt.title(f'MALA Samples at Iteration {iteration}')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Initialize samples from a Gaussian distribution
np.random.seed(42)
num_samples = 100
samples = np.random.randn(num_samples, 2)

# Perform MALA sampling
num_iterations = 1000
step_size = 2.5
mala_samples = []

current_position = samples[0]

# Sampling loop
for i in range(num_iterations):
    current_position = mala_step(current_position, grad_log_p, step_size)
    mala_samples.append(current_position)
    if i % 100 == 0 or i == num_iterations - 1:
        plot_particles(np.array(mala_samples), target_distribution, i + 1)

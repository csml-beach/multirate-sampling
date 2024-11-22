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

# Function to plot particles and contours of the distribution
def plot_particles(particles, target_distribution, iteration):
    plt.figure(figsize=(8, 6))
    # Plot the target distribution
    x, y = np.mgrid[-5:5:.01, -5:5:.01]
    pos = np.dstack((x, y))
    plt.contourf(x, y, target_distribution(pos), levels=30, cmap='viridis', alpha=0.6)
    
    # Plot the particles
    plt.scatter(particles[:, 0], particles[:, 1], color='red', s=10)
    plt.title(f'Gibbs Sampling Particles at Iteration {iteration}')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Gibbs sampling function with plotting
def gibbs_sampling(n_samples, initial_position):
    samples = np.zeros((n_samples, 2))
    samples[0] = initial_position
    
    for i in range(1, n_samples):
        x_prev = samples[i-1]
        
        # Sample x given y
        x_cond_mean = 2 if x_prev[1] > 0 else -2
        x_cond_var = 0.5
        samples[i, 0] = np.random.normal(x_cond_mean, np.sqrt(x_cond_var))
        
        # Sample y given x
        y_cond_mean = 2 if samples[i, 0] > 0 else -2
        y_cond_var = 0.5
        samples[i, 1] = np.random.normal(y_cond_mean, np.sqrt(y_cond_var))
        
        # Plot every 100 samples
        if i % 100 == 0:
            plot_particles(samples[:i], target_distribution, i)
    
    return samples

# Parameters for Gibbs sampling
n_samples = 1000
initial_position = np.array([0, 0])

# Generate samples using Gibbs sampling
samples = gibbs_sampling(n_samples, initial_position)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import rbf_kernel
from scipy.integrate import odeint

# Define the target distribution: a 2D Gaussian mixture
def target_distribution(x):
    # Mixture of two Gaussians
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

# Function to compute the fast dynamics for odeint
def fast_dynamics(particles_flattened, t, kernel_grad):
    particles = particles_flattened.reshape(-1, 2)
    svgd_direction_fast = kernel_grad.sum(axis=0) / len(particles)
    return svgd_direction_fast.flatten()

# SVGD Update Function with MIS using odeint for fast dynamics
def svgd_update(particles, grad_log_p, kernel_bandwidth=0.5, step_size=0.1):
    # Compute the kernel matrix and gradients
    kernel_matrix = rbf_kernel(particles, particles, gamma=1 / (2 * kernel_bandwidth**2))
    kernel_grad = - (particles[:, None, :] - particles[None, :, :]) * kernel_matrix[:, :, None] / kernel_bandwidth**2
    
    # Compute the slow part of the SVGD direction
    grad_log_p_values = np.array([grad_log_p(x) for x in particles])
    svgd_direction_slow = kernel_matrix @ grad_log_p_values / len(particles)
    
    # Integrate the fast dynamics using odeint
    t = np.linspace(0, step_size, 10)
    particles_flattened = particles.flatten()
    particles_flattened = odeint(fast_dynamics, particles_flattened, t, args=(kernel_grad,))
    
    # Update particles with the integrated fast dynamics and slow dynamics
    particles_flattened = particles_flattened[-1]
    particles = particles_flattened.reshape(-1, 2)
    
    svgd_direction_fast = kernel_grad.sum(axis=0) / len(particles)
    svgd_direction = svgd_direction_slow + svgd_direction_fast
    
    particles += step_size * svgd_direction / len(particles)
    
    return particles

# Visualization function
def plot_particles(particles, target_distribution, iteration):
    plt.figure(figsize=(8, 6))
    # Plot the target distribution
    x, y = np.mgrid[-5:5:.01, -5:5:.01]
    pos = np.dstack((x, y))
    plt.contourf(x, y, target_distribution(pos), levels=30, cmap='viridis', alpha=0.6)
    
    # Plot the particles
    plt.scatter(particles[:, 0], particles[:, 1], color='red', s=10)
    plt.title(f'SVGD Particles at Iteration {iteration}')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Initialize particles from a Gaussian distribution
np.random.seed(42)
num_particles = 100
particles = np.random.randn(num_particles, 2)

# Perform SVGD updates and plot
num_iterations = 500
for i in range(num_iterations):
    particles = svgd_update(particles, grad_log_p)
    if i % 20 == 0 or i == num_iterations - 1:
        plot_particles(particles, target_distribution, i + 1)

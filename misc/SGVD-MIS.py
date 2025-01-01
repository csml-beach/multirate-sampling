import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import rbf_kernel

# #Define the target distribution: a 2D Gaussian mixture
# def target_distribution(x):
#     # Mixture of two Gaussians
#     mean1, cov1 = np.array([2, 2]), np.array([[0.5, 0.1], [0.1, 0.5]])
#     mean2, cov2 = np.array([-2, -2]), np.array([[0.5, -0.1], [-0.1, 0.5]])
#     pdf1 = multivariate_normal.pdf(x, mean=mean1, cov=cov1)
#     pdf2 = multivariate_normal.pdf(x, mean=mean2, cov=cov2)
#     return 0.5 * pdf1 + 0.5 * pdf2

# # Gradient of the log target distribution
# def grad_log_p(x):
#     mean1, cov1 = np.array([2, 2]), np.array([[0.5, 0.1], [0.1, 0.5]])
#     mean2, cov2 = np.array([-2, -2]), np.array([[0.5, -0.1], [-0.1, 0.5]])
    
#     pdf1 = multivariate_normal.pdf(x, mean=mean1, cov=cov1)
#     pdf2 = multivariate_normal.pdf(x, mean=mean2, cov=cov2)
    
#     grad1 = -np.linalg.inv(cov1) @ (x - mean1)
#     grad2 = -np.linalg.inv(cov2) @ (x - mean2)
    
#     grad = 0.5 * (grad1 * pdf1 + grad2 * pdf2) / (pdf1 + pdf2)
#     return grad

# # Define the target distribution: Ring distribution
# def target_distribution(x):
#     radius = 3
#     return np.exp(-0.5 * ((np.linalg.norm(x, axis=2) - radius) ** 2))

# def grad_log_p(x):
#     # Ensure x is a 2D array with shape (n, 2)
#     if x.ndim == 1:
#         x = x.reshape(-1, 2)
#     elif x.shape[1] != 2:
#         raise ValueError("Input array must have shape (n, 2)")

#     radius = 3
#     norm_x = np.linalg.norm(x, axis=1, keepdims=True)
#     return - (norm_x - radius) * x / norm_x


# # # Define the target distribution: Banana distribution
# def target_distribution(x):
#     x1, x2 = x[:,:, 0], x[:,:, 1]
#     return np.exp(-0.5 * ((x1 ** 2 / 100) + (x2 + 0.03 * x1 ** 2 - 3) ** 2))

# def grad_log_p(x):
#     if x.ndim == 1:
#         x = x.reshape(1, -1)
#     x1, x2 = x[:, 0], x[:, 1]
#     grad_x1 = -x1 / 50 - 0.06 * x1 * (x2 + 0.03 * x1 ** 2 - 3)
#     grad_x2 = -(x2 + 0.03 * x1 ** 2 - 3)
#     return np.column_stack((grad_x1, grad_x2))

# Define the target distribution: Squiggly distribution
def target_distribution(x):
    x1, x2 = x[:,:, 0], x[:,:, 1]
    return np.exp(-0.5 * ((x2 - np.sin(3 * x1)) ** 2 / 0.1))

def grad_log_p(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x1, x2 = x[:, 0], x[:, 1]
    grad_x1 = -3 * np.cos(3 * x1) * (x2 - np.sin(3 * x1)) / 0.1
    grad_x2 = -(x2 - np.sin(3 * x1)) / 0.1
    return np.column_stack((grad_x1, grad_x2))

# def target_distribution(x):
#     return ((np.floor(x[:,:, 0]) + np.floor(x[:,:, 1])) % 2 == 0).astype(float)

# def grad_log_p(x):
#     return np.zeros_like(x)


# def target_distribution(x):
#     theta = np.arctan2(x[:,:, 1], x[:,:, 0])
#     r = np.sqrt(x[:,:, 0]**2 + x[:,:, 1]**2)
#     return np.exp(-0.5 * ((r - theta) ** 2))

# def grad_log_p(x):
#     if x.ndim == 1:
#         x = x.reshape(-1, 2)
#     theta = np.arctan2(x[:, 1], x[:, 0])
#     r = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
#     grad_r = (r - theta) * x / r
#     grad_theta = (r - theta) * np.array([-x[:, 1], x[:, 0]]).T / (r**2)
#     return grad_r + grad_theta

# SVGD Update Function with MIS
def svgd_update(particles, grad_log_p, kernel_bandwidth=0.5, step_size=0.1, fast_steps=10):
    # Compute the kernel matrix and gradients
    kernel_matrix = rbf_kernel(particles, particles, gamma=1 / (2 * kernel_bandwidth**2))
    kernel_grad = - (particles[:, None, :] - particles[None, :, :]) * kernel_matrix[:, :, None] / kernel_bandwidth**2
    
    # Compute the slow part of the SVGD direction
    grad_log_p_values = np.array([grad_log_p(x) for x in particles]).reshape(len(particles), -1)

    # print('shape of grad_log_p_values:', grad_log_p_values.shape, 'shape of kernel_matrix:', kernel_matrix.shape)
    svgd_direction_slow = kernel_matrix @ grad_log_p_values / len(particles)
    
    # Update particles using MIS
    for _ in range(fast_steps):
        svgd_direction_fast = kernel_grad.sum(axis=0) / len(particles)
        svgd_direction = svgd_direction_slow + svgd_direction_fast
        particles += step_size * svgd_direction / fast_steps
    
    return particles

# Visualization function
def plot_particles(particles, target_distribution, iteration):
    plt.figure(figsize=(8, 6))
    # Plot the target distribution
    x, y = np.mgrid[-5:5:.01, -5:5:.01]
    pos = np.dstack((x, y))
    targets = target_distribution(pos)
    # print('shape of targets:', targets.shape, 'shape of x:', x.shape, 'shape of y:', y.shape)
    plt.contourf(x, y, targets , levels=30, cmap='viridis', alpha=0.6)
    
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
num_iterations = 5000
for i in range(num_iterations):
    particles = svgd_update(particles, grad_log_p)
    if i % 1000 == 0 or i == num_iterations - 1:
        plot_particles(particles, target_distribution, i + 1)

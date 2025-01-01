import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
torch.manual_seed(42)
x_data = torch.linspace(-3, 3, 50).unsqueeze(1)  # Features, shape (50, 1)
true_w0, true_w1, true_noise = 1.5, -2.0, 0.2    # True parameters

# Non-linear basis: sin(x) and cos(x)
phi_x = torch.cat([torch.sin(x_data), torch.cos(2*x_data)], dim=1)  # Shape: (50, 2)

# Generate noisy observations
y_data = true_w0 * phi_x[:, 0] + true_w1 * phi_x[:, 1] + true_noise * torch.randn(x_data.shape[0])

# Plot the data
plt.scatter(x_data.squeeze().numpy(), y_data.numpy(), label='Data', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Synthetic Data for Bayesian Non-Linear Regression')
plt.show()


# 2. Define Log-Posterior (Prior + Likelihood)
def log_posterior(weights, x_basis, y, sigma_p=1.0, sigma_l=0.2):
    """
    Log posterior for Bayesian non-linear regression with two parameters.
    Args:
        weights: Tensor of shape (2,) representing [w0, w1].
        x_basis: Transformed basis features, shape (N, 2).
        y: Observations, shape (N,).
        sigma_p: Standard deviation of Gaussian prior.
        sigma_l: Standard deviation of Gaussian likelihood.
    Returns:
        Scalar log-posterior.
    """
    w0, w1 = weights[0], weights[1]
    prior = -0.5 * (w0**2 + w1**2) / sigma_p**2  # Gaussian prior
    predictions = w0 * x_basis[:, 0] + w1 * x_basis[:, 1]
    residuals = y.squeeze() - predictions
    likelihood = -0.5 * torch.sum((residuals / sigma_l)**2)  # Gaussian likelihood
    return prior + likelihood


# 3. SVGD Update Function
def svgd_update(particles, log_posterior, x_basis, y, bandwidth=0.5, step_size=0.01):
    N, D = particles.shape  # N: number of particles, D: dimensions of weights

    # Compute pairwise distances and kernel matrix
    pairwise_distances = torch.cdist(particles, particles) ** 2  # Shape: (N, N)
    kernel_matrix = torch.exp(-pairwise_distances / (2 * bandwidth**2))  # Shape: (N, N)

    # Kernel gradient: Shape (N, N, D)
    kernel_grad = -(particles.unsqueeze(1) - particles.unsqueeze(0)) * kernel_matrix.unsqueeze(-1) / (bandwidth**2)

    # Compute gradients of log posterior
    grad_log_p_values = []
    for particle in particles:
        particle.requires_grad_(True)
        log_p = log_posterior(particle, x_basis, y)
        grad = torch.autograd.grad(log_p, particle, retain_graph=True)[0]
        grad_log_p_values.append(grad.detach())
        particle.requires_grad_(False)
    grad_log_p_values = torch.stack(grad_log_p_values)  # Shape: (N, D)

    # Compute SVGD update direction
    phi = (kernel_matrix @ grad_log_p_values) / N  # Shape: (N, D)
    phi += kernel_grad.sum(dim=0) / N  # Shape: (N, D)

    # Update particles
    particles = particles + step_size * phi
    return particles


# 4. Main Script
if __name__ == "__main__":
    # Non-linear basis transformation: sin(x) and cos(x)
    x_basis = torch.cat([torch.sin(x_data), torch.cos(2*x_data)], dim=1)  # Shape: (N, 2)

    # Initialize particles for weights w0 and w1
    num_particles = 100
    particles = torch.randn(num_particles, 2)  # Each particle has two dimensions: [w0, w1]

    # Run SVGD and track particle history
    num_iterations = 100
    all_particles = [particles.clone()]

    for i in range(num_iterations):
        particles = svgd_update(particles, log_posterior, x_basis, y_data)
        all_particles.append(particles.clone())

    all_particles = torch.stack(all_particles)

    # 5. Joint Distribution of w0 and w1
    w0_samples, w1_samples = particles[:, 0], particles[:, 1]
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=w0_samples.numpy(), y=w1_samples.numpy(), cmap="Blues", fill=True, levels=30, thresh=0.05)
    plt.scatter(w0_samples.numpy(), w1_samples.numpy(), color='red', s=10, alpha=0.5, label='Particles')
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.title('Joint Distribution of w0 and w1')
    plt.legend()
    plt.show()

    # 6. Predictive Distribution
    predictions = []
    for p in particles:
        w0, w1 = p[0], p[1]
        predictions.append(w0 * x_basis[:, 0] + w1 * x_basis[:, 1])
    
    predictions = torch.stack(predictions)
    mean_prediction = predictions.mean(dim=0)
    std_prediction = predictions.std(dim=0)

    # Plot predictive mean and uncertainty
    plt.scatter(x_data.squeeze().numpy(), y_data.squeeze().numpy(), label='Data', color='blue')
    plt.plot(x_data.squeeze().numpy(), mean_prediction.numpy(), label='Predictive Mean', color='red')
    plt.fill_between(
        x_data.squeeze().numpy(),
        (mean_prediction - std_prediction).numpy(),
        (mean_prediction + std_prediction).numpy(),
        color='red', alpha=0.2, label='Uncertainty'
    )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Predictive Distribution with Uncertainty')
    plt.show()

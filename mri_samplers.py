import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import torchdiffeq
from abc import ABC, abstractmethod


class TargetDistribution:
    def __init__(self, target, device):
        self.target = target
        self.device = device
        self.distributions = {
            'spirals': (self.spirals_distribution, self.spirals_grad_log_p),
            'squiggly': (self.squiggly_distribution, self.squiggly_grad_log_p),
            'banana': (self.banana_distribution, self.banana_grad_log_p),
            'ring': (self.ring_distribution, self.ring_grad_log_p),
            'gaussian': (self.gaussian_distribution, self.gaussian_grad_log_p),
            'checkerboard': (self.checkerboard_distribution, self.checkerboard_grad_log_p)
        }

        if target not in self.distributions:
            raise ValueError("Invalid target distribution")

        self.target_distribution, self.grad_log_p = self.distributions[target]

    def spirals_distribution(self, pos):
        x = pos.reshape(-1, 2)
        theta = torch.atan2(x[:, 1], x[:, 0]) + 2 * torch.pi
        r = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
        density = torch.exp(-4 * 0.5 * ((r - (theta / (2 * torch.pi) * 4)) ** 2))  # Decreased radius
        return density.reshape(pos.shape[:-1])

    def spirals_grad_log_p(self, x):
        theta = torch.atan2(x[:, 1], x[:, 0]) + 2 * torch.pi
        r = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
        grad_r = (r - (theta / (2 * torch.pi) * 4)) * x / r.unsqueeze(1)  # Decreased radius
        grad_theta = (r - (theta / (2 * torch.pi) * 4)) * torch.stack([-x[:, 1], x[:, 0]], dim=-1) / (r.unsqueeze(1)**2)
        return -4 * (grad_r + grad_theta)

    def squiggly_distribution(self, x):
        x1, x2 = x[:,:, 0], x[:,:, 1]
        return torch.exp(-0.25 * ((x2 - torch.sin(3 * x1)) ** 2 / 0.1))

    def squiggly_grad_log_p(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x1, x2 = x[:, 0], x[:, 1]
        grad_x1 = -3 * torch.cos(3 * x1) * (x2 - torch.sin(3 * x1)) / 0.1
        grad_x2 = -(x2 - torch.sin(3 * x1)) / 0.1
        return  0.5 * torch.column_stack((grad_x1, grad_x2))

    def banana_distribution(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        return torch.exp(-2.0 * ((x1**2 / 100) + (x2 + 0.03 * x1**2 - 3) ** 2))

    def banana_grad_log_p(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        grad_x1 = -x1 / 50 - 0.06 * x1 * (x2 + 0.03 * x1**2 - 3)
        grad_x2 = -(x2 + 0.03 * x1**2 - 3)
        return torch.stack((grad_x1, grad_x2), dim=-1)

    def ring_distribution(self, x):
        radius = 4
        norm_x = torch.norm(x, dim=-1)
        return torch.exp(-2.0 * (norm_x - radius) ** 2)

    def ring_grad_log_p(self, x):
        radius = 4
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        return -4.0 * (norm_x - radius) * x / norm_x

    def gaussian_distribution(self, x):
        if x.device.type == 'mps':
            x = x.cpu()
        mean1, cov1 = torch.tensor([2.0, 2.0]), torch.tensor([[0.5, 0.1], [0.1, 0.5]])
        mean2, cov2 = torch.tensor([-2.0, -2.0]), torch.tensor([[0.5, -0.1], [-0.1, 0.5]])
        pdf1 = torch.exp(-0.5 * torch.sum((x - mean1) @ torch.linalg.inv(cov1) * (x - mean1), dim=-1)) / torch.sqrt(torch.det(cov1) * (2 * torch.pi) ** 2)
        pdf2 = torch.exp(-0.5 * torch.sum((x - mean2) @ torch.linalg.inv(cov2) * (x - mean2), dim=-1)) / torch.sqrt(torch.det(cov2) * (2 * torch.pi) ** 2)
        pdf1 = pdf1.to(self.device)
        pdf2 = pdf2.to(self.device)
        return 0.5 * pdf1 + 0.5 * pdf2

    def gaussian_grad_log_p(self, x):
        if x.device.type == 'mps':
            x = x.cpu()
        mean1, cov1 = torch.tensor([2.0, 2.0]), torch.tensor([[0.5, 0.1], [0.1, 0.5]])
        mean2, cov2 = torch.tensor([-2.0, -2.0]), torch.tensor([[0.5, -0.1], [-0.1, 0.5]])
        pdf1 = torch.exp(-0.5 * torch.sum((x - mean1) @ torch.linalg.inv(cov1) * (x - mean1), dim=-1)) / torch.sqrt(torch.det(cov1) * (2 * torch.pi) ** 2)
        pdf2 = torch.exp(-0.5 * torch.sum((x - mean2) @ torch.linalg.inv(cov2) * (x - mean2), dim=-1)) / torch.sqrt(torch.det(cov2) * (2 * torch.pi) ** 2)
        grad1 = -(x - mean1) @ torch.linalg.inv(cov1).T
        grad2 = -(x - mean2) @ torch.linalg.inv(cov2).T
        d_log_p = 0.5 * (grad1 * pdf1[:, None] + grad2 * pdf2[:, None]) / (pdf1 + pdf2).unsqueeze(-1)
        d_log_p = d_log_p.to(self.device)
        return d_log_p

    def checkerboard_distribution(self, x):
        device = x.device
        x1, x2 = x[:,:, 0], x[:,:, 1]
        scale = 3.0  # Adjust this scale to control the size of the squares
        checkered = ((torch.floor(x1 / scale) + torch.floor(x2 / scale)) % 2 == 0).float()
        return checkered

    def checkerboard_grad_log_p(self, x):
        # Gradient is zero everywhere for a checkerboard pattern as it is piecewise constant
        return torch.zeros_like(x)

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import torchdiffeq
from abc import ABC, abstractmethod


class SamplingMethod(ABC):
    def __init__(self, particles, grad_log_p, target_distribution, t_final, bandwidth=0.5, counter=None):
        self.particles = particles
        self.grad_log_p = grad_log_p
        self.target_distribution = target_distribution
        self.t_final = t_final
        self.bandwidth = bandwidth
        self.counter = counter

    @abstractmethod
    def run(self):
        pass


class SVGDSampler(SamplingMethod):
    def run(self, metrics_tracker):
        # Implement SVGD logic here
        all_particles = []
        step_size = 0.1
        t = torch.tensor(0.0, device=self.particles.device)

        while t < self.t_final:
            if t + step_size > self.t_final:
                step_size = self.t_final - t

            metrics_tracker.track(self.particles, self.target_distribution, self.counter.get_flops(), t.item())
            distances = torch.cdist(self.particles, self.particles) ** 2
            self.counter.count_subtraction(self.particles.unsqueeze(1), self.particles.unsqueeze(0))
            self.counter.count_exp(distances)
            kernel_matrix = torch.exp(-distances / (2 * self.bandwidth**2))
            self.counter.count_elementwise_div(kernel_matrix, 2 * self.bandwidth**2)

            grad_log_p_values = self.grad_log_p(self.particles)
            self.counter.count_matmul(kernel_matrix, grad_log_p_values)
            svgd_direction = kernel_matrix @ grad_log_p_values / len(self.particles)
            self.counter.count_elementwise_div(svgd_direction, len(self.particles))

            self.particles = self.particles + step_size * svgd_direction
            self.counter.count_elementwise_mul(svgd_direction, step_size)
            self.counter.count_addition(self.particles, svgd_direction)

            t = t + step_size
            all_particles.append(self.particles.clone())

        return all_particles


class MRISampler(SamplingMethod):
    def run(self, metrics_tracker):
        # Implement multirate SVGD logic here
        all_particles = []
        step_size = 0.1
        t = torch.tensor(0.0, device=self.particles.device)

        while t < self.t_final:
            if t + step_size > self.t_final:
                step_size = self.t_final - t

            distances = torch.cdist(self.particles, self.particles) ** 2
            self.counter.count_subtraction(self.particles.unsqueeze(1), self.particles.unsqueeze(0))
            self.counter.count_exp(distances)
            kernel_matrix = torch.exp(-distances / (2 * self.bandwidth**2))
            self.counter.count_elementwise_div(kernel_matrix, 2 * self.bandwidth**2)

            def f_fast(t, particles):
                kernel_grad = -(particles.unsqueeze(1) - particles.unsqueeze(0)) * kernel_matrix.unsqueeze(-1) / (self.bandwidth**2)
                return 1 * kernel_grad.sum(dim=0) / len(particles)

            def f_slow(t, particles):
                grad_log_p_values = self.grad_log_p(particles)
                return kernel_matrix @ grad_log_p_values / len(particles)

            ynext, yhat = mriGARKstep(self.particles, f_slow, f_fast, step_size)
            relative_error = torch.norm(ynext - yhat) / torch.clamp(torch.norm(ynext), min=1e-6)

            if relative_error > 0.5:
                step_size = 0.8 * step_size
                continue
            else:
                t = t + step_size
                step_size = step_size * 1.2
                self.particles = ynext.clone()
                all_particles.append(ynext)
                metrics_tracker.track(self.particles, self.target_distribution, self.counter.get_flops(), t.item())

        return all_particles


def mriGARKstep(particles, f_slow, f_fast, step_size):
    t = torch.tensor(0.0).to(particles.device)
    v1 = particles.clone() 
    fs = f_slow(0, particles)
    stage_ode = lambda t, v: 0.5 * f_fast(t, v) + 0.5 * fs
    v2 = torchdiffeq.odeint(stage_ode, v1, torch.tensor([0, step_size], device=particles.device))[-1,:,:]

    fs2 = f_slow(0, v2)
    stage_ode = lambda t, v: 0.5 * f_fast(t, v) - 0.5 * fs + fs2
    ynext = torchdiffeq.odeint(stage_ode, v2, torch.tensor([0, step_size], device=particles.device))[-1,:,:]

    v3 = v2
    stage_ode = lambda t, v: 0.5 * f_fast(t, v) + 0.5 * fs
    yhat = torchdiffeq.odeint(stage_ode, v3, torch.tensor([0, step_size], device=particles.device))[-1,:,:]
    
    return ynext, yhat


class FlopCounter:
    def __init__(self):
        self.flops = 0  # Initialize total FLOP count

    def count_matmul(self, A, B):
        m, n = A.shape
        n, p = B.shape
        self.flops += 2 * m * n * p

    def count_addition(self, A, B):
        self.flops += A.numel()

    def count_subtraction(self, A, B):
        self.flops += A.numel()

    def count_elementwise_mul(self, A, B):
        self.flops += A.numel()

    def count_elementwise_div(self, A, B):
        self.flops += A.numel()

    def count_exp(self, A):
        self.flops += A.numel()

    def get_flops(self):
        return self.flops


class SamplingExperiment:
    def __init__(self, target_distribution, grad_log_p, particles, t_final, bandwidth=0.5):
        self.target_distribution = target_distribution
        self.grad_log_p = grad_log_p
        self.particles = particles
        self.t_final = t_final
        self.bandwidth = bandwidth
        self.methods = []

    def add_method(self, method):
        self.methods.append(method)

    def run_all(self):
        results = {}
        for method in self.methods:
            results[type(method).__name__] = method.run(metrics_tracker_svgd if isinstance(method, SVGDSampler) else metrics_tracker_mri)
        return results


class MetricsTracker:
    def __init__(self):
        self.history = {
            'flop_history': [],
            'time_history': [],
            'ess_history': [],
            'mlp_history': [],
            'kl_history': [],
            'spread_history': []
        }

    def track(self, particles, target_distribution, flop_count, time_elapsed):
        # Add metrics tracking logic here
        mlp    = self.mean_log_p(particles, target_distribution)
        ess    = self.effective_sample_size(particles, target_distribution)
        kl     = self.kl_divergence(particles, target_distribution)
        spread = self.compute_sample_spread(particles)

        self.history['flop_history'].append(flop_count)
        self.history['time_history'].append(time_elapsed)
        self.history['mlp_history'].append(mlp)
        self.history['ess_history'].append(ess)
        self.history['kl_history'].append(kl)
        self.history['spread_history'].append(spread)

    def mean_log_p(self, particles, target_distribution):
        log_p = target_distribution(particles.unsqueeze(0)).squeeze()
        return log_p.mean().item()

    def effective_sample_size(self, particles, target_distribution):
        weights = target_distribution(particles.unsqueeze(0)).squeeze()
        normalized_weights = weights / weights.sum()
        ess = 1.0 / torch.sum(normalized_weights ** 2)
        return ess.item()

    def kl_divergence(self, particles, target_distribution, bandwidth=0.5):
        device = particles.device
        N, D = particles.shape
        distances = torch.cdist(particles, particles) ** 2
        kernel_matrix = torch.exp(-distances / (2 * bandwidth**2))
        particle_density = kernel_matrix.sum(dim=1) / (N * (2 * torch.pi * bandwidth**2) ** (D / 2))
        target_density = target_distribution(particles.unsqueeze(0)).squeeze().clamp(min=1e-12)
        kl = torch.mean(torch.log(particle_density) - torch.log(target_density))
        return kl.item()

    def compute_sample_spread(self, samples):
        pairwise_distances = torch.cdist(samples, samples, p=2) ** 2
        num_samples = samples.size(0)
        spread = pairwise_distances.sum() / (num_samples * (num_samples - 1))
        return spread.item()


# Configuration
target = 'squiggly'  # Choose from 'gaussian', 'ring', 'banana', 'squiggly', 'checkerboard', 'spirals'
device = torch.device("mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available() 
                      else "cpu")
device = torch.device('cpu')
print(device)

# Instantiate the target distribution
target_dist = TargetDistribution(target, device)

target_distribution = target_dist.target_distribution
grad_log_p = target_dist.grad_log_p

# Initialize particles
num_particles = 100
particles = (torch.rand(num_particles, 2) - 0.5) * 10
particles = particles.to(device)
t_final = 100.0

# Instantiate flop counters
flop_counter_svgd = FlopCounter()
flop_counter_mri = FlopCounter()

# Instantiate metrics tracker
metrics_tracker_svgd = MetricsTracker()
metrics_tracker_mri = MetricsTracker()

# Set up the experiment
experiment = SamplingExperiment(
    target_distribution=target_distribution,
    grad_log_p=grad_log_p,
    particles=particles,
    t_final=t_final,
    bandwidth=0.5
)

# Add different sampling methods
svgd_sampler = SVGDSampler(particles, grad_log_p, target_distribution, t_final, bandwidth=0.5, counter=flop_counter_svgd)
mri_sampler = MRISampler(particles, grad_log_p, target_distribution, t_final, bandwidth=0.5, counter=flop_counter_mri)

experiment.add_method(svgd_sampler)
experiment.add_method(mri_sampler)

# Run all methods
results = experiment.run_all()


# # Track metrics for each step
# for method_name, particles_list in results.items():
#     for i, particles in enumerate(particles_list):
#         if method_name == 'SVGDSampler':
#             metrics_tracker_svgd.track(particles, target_distribution, flop_counter_svgd.get_flops(), i)
#         elif method_name == 'MRISampler':
#             metrics_tracker_mri.track(particles, target_distribution, flop_counter_mri.get_flops(), i)

# Plotting results
fig = plt.figure(figsize=(16, 4))
axes = [fig.add_subplot(1, 4, i + 1) for i in range(4)]

for experiment_name, metrics_tracker in zip(['MRI', 'SVGD'], [metrics_tracker_mri, metrics_tracker_svgd]):
    # Plot the quality metric (mean log-probability
    axes[0].semilogy(metrics_tracker.history['time_history'], metrics_tracker.history['mlp_history'], label=experiment_name)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Mean Log-Probability')

    # Plot the effective sample size over time
    axes[1].plot(metrics_tracker.history['time_history'], metrics_tracker.history['ess_history'], label=experiment_name)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Effective Sample Size')

    # Plot the KL divergence over time
    axes[2].plot(metrics_tracker.history['time_history'], metrics_tracker.history['kl_history'], label=experiment_name)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('KL Divergence')

    # Plot the spread over time
    axes[3].plot(metrics_tracker.history['time_history'], metrics_tracker.history['spread_history'], label=experiment_name)
    axes[3].set_xlabel('Time')
    axes[3].set_ylabel('Spread')

# Turn on the legend for all subplots
for ax in axes:
    ax.legend()

plt.show()

# Create separate animations for each method
for experiment_name, particles_list in results.items():
    fig, ax = plt.subplots(figsize=(4, 4))

    # Define the grid for the target distribution plot
    x, y = torch.meshgrid(torch.linspace(-5, 5, 100), torch.linspace(-5, 5, 100))
    pos = torch.stack((x, y), dim=-1).to(device)
    target_density = target_distribution(pos).cpu().numpy()

    # Plot the target distribution
    ax.contourf(x.cpu().numpy(), y.cpu().numpy(), target_density, levels=30, cmap='viridis', alpha=0.6)

    # Initialize the scatter plot for particles
    scatter = ax.scatter([], [], color='red', s=10)

    # Update function for animation
    def update(frame):
        scatter.set_offsets(particles_list[frame].cpu().numpy())
        ax.set_title(f"{experiment_name} - Iteration {frame}")
        return scatter,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(particles_list), blit=True)

    # Display the animation
    HTML(ani.to_jshtml())
    plt.show()

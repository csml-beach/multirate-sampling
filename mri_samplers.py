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
            'checkerboard': (self.checkerboard_distribution, self.checkerboard_grad_log_p),
            'swiss': (self.curved_gaussian_mixture, self.curved_gaussian_mixture_grad_log_p)
        }

        if target not in self.distributions:
            raise ValueError("Invalid target distribution")

        self.target_distribution, self.grad_log_p = self.distributions[target]
    

    def curved_gaussian_mixture(self, x, mu1=torch.tensor([-2.0, -2.0]), mu2=torch.tensor([2.0, 2.0]), 
                                sigma1=1.0, sigma2=1.0, weight1=0.5, weight2=0.5, curve_strength=4.5):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x1, x2 = x[..., 0], x[..., 1]

        # Apply a curving transformation (sinusoidal)
        x2_curved = x2 + curve_strength * torch.sin(x1)

        # Compute Gaussian components
        gauss1 = torch.exp(-0.5 * (((x1 - mu1[0]) / sigma1)**2 + ((x2_curved - mu1[1]) / sigma1)**2)) / (2 * torch.pi * sigma1**2)
        gauss2 = torch.exp(-0.5 * (((x1 - mu2[0]) / sigma2)**2 + ((x2_curved - mu2[1]) / sigma2)**2)) / (2 * torch.pi * sigma2**2)

        # Weighted sum of components
        return weight1 * gauss1 + weight2 * gauss2


    def curved_gaussian_mixture_grad_log_p(self, x, mu1=torch.tensor([-2.0, -2.0]), mu2=torch.tensor([2.0, 2.0]), 
                                        sigma1=1.0, sigma2=1.0, weight1=0.5, weight2=0.5, curve_strength=4.5):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x1, x2 = x[..., 0], x[..., 1]

        # Apply a curving transformation
        x2_curved = x2 + curve_strength * torch.sin(x1)
        dx2_dcurve = 1 + curve_strength * torch.cos(x1)

        # Compute Gaussian components
        gauss1 = torch.exp(-0.5 * (((x1 - mu1[0]) / sigma1)**2 + ((x2_curved - mu1[1]) / sigma1)**2)) / (2 * torch.pi * sigma1**2)
        gauss2 = torch.exp(-0.5 * (((x1 - mu2[0]) / sigma2)**2 + ((x2_curved - mu2[1]) / sigma2)**2)) / (2 * torch.pi * sigma2**2)
        total_density = weight1 * gauss1 + weight2 * gauss2
        # Add stabilization to avoid division by zero
        total_density = torch.clamp(total_density, min=1e-9)

        # Gradients for each component
        grad_x1_1 = -((x1 - mu1[0]) / sigma1**2) * gauss1
        grad_x2_1 = -((x2_curved - mu1[1]) / sigma1**2) * dx2_dcurve * gauss1

        grad_x1_2 = -((x1 - mu2[0]) / sigma2**2) * gauss2
        grad_x2_2 = -((x2_curved - mu2[1]) / sigma2**2) * dx2_dcurve * gauss2

        grad_x1 = (weight1 * grad_x1_1 + weight2 * grad_x1_2) / total_density
        grad_x2 = (weight1 * grad_x2_1 + weight2 * grad_x2_2) / total_density

        return torch.stack((grad_x1, grad_x2), dim=-1)

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
    def run(self, metrics_tracker, step_size=0.05):
        all_particles = []
        t = torch.tensor(0.0, device=self.particles.device)
        num_particles = self.particles.size(0)

        while t < self.t_final:
            if t + step_size > self.t_final:
                step_size = self.t_final - t

            # Compute the kernel matrix (RBF kernel)
            pairwise_distances = torch.cdist(self.particles, self.particles) ** 2
            kernel_matrix = torch.exp(-pairwise_distances / (2 * self.bandwidth**2))

            # Compute the gradient of the kernel with respect to particles
            kernel_grad = -(self.particles.unsqueeze(1) - self.particles.unsqueeze(0)) * kernel_matrix.unsqueeze(-1) / (self.bandwidth**2)

            # Compute the gradient of the log-probability for all particles
            grad_log_p_values = self.grad_log_p(self.particles)
            # alternatively, approximate the gradient of the log-probability using kernel gradients
            # grad_log_p_values = kernel_grad.sum(dim=1) / num_particles

            # Compute the SVGD update direction
            phi = (kernel_matrix @ grad_log_p_values) / num_particles + kernel_grad.sum(dim=0) / num_particles

            # Update the particles
            self.particles = self.particles + step_size * phi

            t  = t + step_size

            metrics_tracker.track(self.particles, self.target_distribution, self.counter.get_flops(), t.item())
            all_particles.append(self.particles.clone())

            self.counter.count_addition(self.particles.unsqueeze(1), self.particles.unsqueeze(0))
            self.counter.count_exp(kernel_matrix)
            self.counter.count_matmul(kernel_matrix, grad_log_p_values)
            self.counter.count_elementwise_mul(phi, step_size)
            self.counter.count_addition(self.particles, phi)

        return all_particles

class MRISampler(SamplingMethod):
    def run(self, metrics_tracker, step_size=0.01):
        all_particles = []
        t = torch.tensor(0.0, device=self.particles.device)

        while t < self.t_final:
            if t + step_size > self.t_final:
                step_size = self.t_final - t

            kernel_matrix = compute_kernel_matrix(self.particles, self.bandwidth)
            
            self.counter.count_addition(self.particles.unsqueeze(1), self.particles.unsqueeze(0))
            self.counter.count_exp(kernel_matrix)
            self.counter.count_elementwise_mul(kernel_matrix, 2 * self.bandwidth**2)

            def f_fast(t, particles):
                kernel_grad = -(particles.unsqueeze(1) - particles.unsqueeze(0)) * kernel_matrix.unsqueeze(-1) / (self.bandwidth**2)
                return kernel_grad.sum(dim=0) / len(particles)

            def f_slow(t, particles):
                grad_log_p_values = self.grad_log_p(particles)
                return kernel_matrix @ grad_log_p_values / len(particles)

            step_ratio = 10
            ynext, yhat = mriGARKstep(self.particles, f_slow, f_fast, step_size)
            relative_error = torch.norm(ynext - yhat) / torch.clamp(torch.norm(ynext), min=1e-6)

            if relative_error > 0.01:
                step_size *= 0.8
                continue
            else:
                t += step_size
                step_size *= 1.2
                self.particles = ynext.clone()
                all_particles.append(ynext)
                metrics_tracker.track(self.particles, self.target_distribution, self.counter.get_flops(), t.item())

        return all_particles

class MRSampler(SamplingMethod):
    def run(self, metrics_tracker, step_size=0.05):
        all_particles = []
        t = torch.tensor(0.0, device=self.particles.device)
        num_particles = self.particles.size(0)
        bandwidth = self.bandwidth

        def update_repulsive_dynamics(t, particles):
                
                pairwise_distances = torch.cdist(particles, particles) ** 2
                kernel_matrix = torch.exp(-pairwise_distances / (2 *bandwidth**2))
                kernel_grad = -(particles.unsqueeze(1) - particles.unsqueeze(0)) * kernel_matrix.unsqueeze(-1) / (bandwidth**2)
                update = kernel_grad.sum(dim=0) / num_particles

                return particles + step_size * update
            
        def update_attractive_dynamics(t, particles):
                pairwise_distances = torch.cdist(particles, particles) ** 2
                kernel_matrix = torch.exp(-pairwise_distances / (2 * bandwidth**2))
                grad_log_p_values = self.grad_log_p(particles)
                update = kernel_matrix @ grad_log_p_values / num_particles
                return particles + step_size * update

        while t < self.t_final:
            if t + step_size > self.t_final:
                step_size = self.t_final - t
            
            self.particles = update_repulsive_dynamics(t, self.particles)

            

            for _ in range(10):
                self.particles = update_attractive_dynamics(t, self.particles)
                t = t + step_size

            metrics_tracker.track(self.particles, self.target_distribution, self.counter.get_flops(), t.item())
            all_particles.append(self.particles.clone())

            # self.counter.count_addition(self.particles.unsqueeze(1), self.particles.unsqueeze(0))
            # self.counter.count_exp(kernel_matrix)
            # self.counter.count_matmul(kernel_matrix, grad_log_p_values)
            # self.counter.count_elementwise_mul(ynext, step_size)
            # self.counter.count_addition(self.particles, ynext)

        return all_particles
    
def mriGARKstep(particles, f_slow, f_fast, step_size):
    t  = torch.tensor(0.0).to(particles.device)
    v1 = particles.clone() 
    fs = f_slow(0, particles)
    stage_ode = lambda t, v: 0.5 * f_fast(t, v) + 0.5 * fs
    ode_options = {'min_step': 1e-3}

    v2 = torchdiffeq.odeint(stage_ode, v1, torch.tensor([0, step_size], device=particles.device),
                            rtol=1e-3, atol=1e-3,
                            options=ode_options)[-1,:,:]

    fs2 = f_slow(0, v2)
    stage_ode = lambda t, v: 0.5 * f_fast(t, v) - 0.5 * fs + fs2
    ynext = torchdiffeq.odeint(stage_ode, v2, torch.tensor([0, step_size], device=particles.device),
                               rtol=1e-3, atol=1e-3,
                               options=ode_options)[-1,:,:]

    v3 = v2
    stage_ode = lambda t, v: 0.5 * f_fast(t, v) + 0.5 * fs
    yhat = torchdiffeq.odeint(stage_ode, v3, torch.tensor([0, step_size], device=particles.device),
                              rtol=1e-3, atol=1e-3,
                              options=ode_options)[-1,:,:]
    
    return ynext, yhat


def mrStep(particles, f_slow, f_fast, step_size, step_ratio):

    # Initialize time and particles
    t = torch.tensor(0.0).to(particles.device)
    ynext = particles.clone()

    fbefore = f_slow(t, particles) + f_fast(t, particles)

    
    # Compute slow dynamics
    fs = f_slow(t, particles)
    
    ynext += step_size * fs

    # Perform step_ratio fast updates for one slow update
    fast_step_size = step_size / step_ratio
    for _ in range(step_ratio):
        f_fast_eval = f_fast(t, ynext)
        ynext += fast_step_size * (f_fast_eval)

    f_after = f_slow(t + step_size, ynext) + f_fast(t + step_size, ynext)

    # Set yhat to be the same as ynext
    growth = torch.norm(f_after)/torch.norm(fbefore)

    return ynext, growth


class FlopCounter:
    def __init__(self):
        self.flops = 0  # Initialize total FLOP count

    def count_matmul(self, A, B):
        m, n = A.shape
        n, p = B.shape
        self.flops += 2 * m * n * p

    def count_addition(self, A, B):
        if A.shape != B.shape:
            self.flops += (A.numel())**2
        else:
            self.flops += A.numel()

    def count_elementwise_mul(self, A, B):
        # note: it is assumed that A and B have the same shape or B is a scalar
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

    def run_all(self, trackers):
        results = {}
        for method, tracker in zip(self.methods, trackers):
            results[type(method).__name__] = method.run(tracker)
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
        # this is in fact mean probability, not mean log probability, we will use semilogy to plot it
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

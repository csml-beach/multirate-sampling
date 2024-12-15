import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from mri_samplers import * 

def setup_experiment(target, device, num_particles, t_final, seed=42):
    torch.manual_seed(seed)
    target_dist = TargetDistribution(target, device)
    particles = (torch.rand(num_particles, 2) - 0.5) * 10
    particles = particles.to(device)
    return target_dist, particles

def plot_metrics(metrics_trackers, labels):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for label, tracker in zip(labels, metrics_trackers):
        axes[0].semilogy(tracker.history['time_history'], tracker.history['mlp_history'], label=label)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Mean Log Probability')
        axes[1].plot(tracker.history['time_history'], tracker.history['ess_history'], label=label)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Effective Sample Size')
        axes[2].plot(tracker.history['time_history'], tracker.history['kl_history'], label=label)
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('KL Divergence')

        axes[3].plot(tracker.history['time_history'], tracker.history['spread_history'], label=label)
        axes[3].set_xlabel('Time')
        axes[3].set_ylabel('Sample Spread')
    for ax in axes:
        ax.legend()
    plt.show()

def animate_results(results, target_distribution, device):
    x, y = torch.meshgrid(torch.linspace(-5, 5, 100), torch.linspace(-5, 5, 100), indexing='xy')
    pos = torch.stack((x, y), dim=-1).to(device)
    target_density = target_distribution(pos).cpu().numpy()

    for experiment_name, particles_list in results.items():
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.contourf(x.cpu().numpy(), y.cpu().numpy(), target_density, levels=30, cmap='seismic', alpha=0.6)
        scatter = ax.scatter([], [], color='red', s=10)
        
        # Filter out None values from particles_list
        particles_np = torch.stack(particles_list).cpu().numpy()

        def update(frame):
            scatter.set_offsets(particles_np[frame])
            ax.set_title(f"{experiment_name} - Iteration {frame}")
            return scatter,

        ani = FuncAnimation(fig, update, frames=len(particles_np), blit=True)
        # HTML(ani.to_html5_video())
        ani.save(f'{experiment_name}.mp4', writer='ffmpeg', fps=60)
        plt.close(fig)

# Configuration
t_final = 100.0
animate = True
plot = True
target = 'squiggly'
# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(device)

# Setup experiment
target_dist, particles = setup_experiment(target, device, num_particles=100,
                                           t_final=t_final, seed=42)
target_distribution = target_dist.target_distribution
grad_log_p = target_dist.grad_log_p

# Instantiate flop counters and metrics trackers
flop_counter_svgd = FlopCounter()
flop_counter_mri = FlopCounter()
flop_count_mr = FlopCounter()
metrics_tracker_svgd = MetricsTracker()
metrics_tracker_mri = MetricsTracker()
metrics_tracker_mr = MetricsTracker()

# Set up the experiment
experiment = SamplingExperiment(target_distribution, grad_log_p, particles,
                                 t_final=t_final, bandwidth=0.5)
svgd_sampler = SVGDSampler(particles.clone(), grad_log_p, target_distribution,
                            t_final=t_final, bandwidth=0.5, counter=flop_counter_svgd)
# mri_sampler = MRISampler(particles.clone(), grad_log_p, target_distribution,
#  t_final=t_final, bandwidth=0.5, counter=flop_counter_mri)
mr_sampler = MRSampler(particles.clone(), grad_log_p, target_distribution,
                        t_final=t_final, bandwidth=0.5, counter=flop_count_mr)

experiment.add_method(svgd_sampler)
experiment.add_method(mr_sampler)

results = experiment.run_all([metrics_tracker_svgd, metrics_tracker_mr])

if plot:
    plot_metrics([metrics_tracker_svgd, metrics_tracker_mr], ['SVGD', 'MRI'])

if animate:
    animate_results(results, target_distribution, device)


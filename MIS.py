import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the ODE system
def f_slow(y1, y2, t):
    """Slow component of the system."""
    return -0.1 * y1 + 2 * y2

def f_fast(y2, t, y1):
    """Fast component of the system."""
    return -50 * (y2 - np.sin(t))

# Multirate Infinitesimal Step (MIS) integrator with scipy.odeint
def mis_integrate_with_odeint(y1_init, y2_init, t_start, t_end, dt_slow, dt_fast):
    """Integrate the system using the MIS method."""
    times = np.arange(t_start, t_end, dt_slow)
    y1 = [y1_init]  # Slow component
    y2 = [y2_init]  # Fast component

    for i, t in enumerate(times[:-1]):
        # Update y1 (slow) using explicit Euler
        y1_new = y1[-1] + dt_slow * f_slow(y1[-1], y2[-1], t)
        
        # Update y2 (fast) using odeint
        fast_times = np.arange(t, t + dt_slow, dt_fast)  # Sub-steps for fast component
        y2_result = odeint(f_fast, y2[-1], fast_times, args=(y1[-1],))
        y2_new = y2_result[-1, 0]  # Take the final value after integration
        
        # Store the new values
        y1.append(y1_new)
        y2.append(y2_new)

    return times, np.array(y1), np.array(y2)

# Initial conditions and parameters
y1_init = 1.0
y2_init = 0.0
t_start = 0.0
t_end = 10.0
dt_slow = 0.1
dt_fast = 0.01

# Solve the system
times, y1, y2 = mis_integrate_with_odeint(y1_init, y2_init, t_start, t_end, dt_slow, dt_fast)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(times, y1, label="y1 (Slow)", color="blue")
plt.plot(times, y2, label="y2 (Fast)", color="red", alpha=0.6)
plt.title("Multirate Infinitesimal Step (MIS) Method with odeint")
plt.xlabel("Time")
plt.ylabel("Values")
plt.legend()
plt.grid()
plt.show()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, lax

# Configuration
L = 100.0       # Domain size
NX, NY = 128, 128
DX = L / NX
DY = L / NY
G = 9.81        # Gravity
DT = 0.01       # Timestep
STEPS = 500

def initialize():
    """Initial condition: A quiet pool with a central Gaussian ripple."""
    x = jnp.linspace(0, L, NX)
    y = jnp.linspace(0, L, NY)
    X, Y = jnp.meshgrid(x, y)
    
    # Static depth (flat bottom)
    H0 = 10.0
    
    # Perturbation (the "ripple")
    eta = 2.0 * jnp.exp(-((X - L/2)**2 + (Y - L/2)**2) / (2 * 5.0**2))
    
    H = H0 + eta
    U = jnp.zeros_like(H)
    V = jnp.zeros_like(H)
    
    return H, U, V

def get_fluxes(H, U, V):
    """Compute the fluxes for the Shallow Water Equations."""
    # Conservative variables: H, HU, HV
    HU = H * U
    HV = H * V
    
    # Fluxes in X
    Fx_H = HU
    Fx_HU = HU**2 / H + 0.5 * G * H**2
    Fx_HV = HU * V
    
    # Fluxes in Y
    Fy_H = HV
    Fy_HU = HV * U
    Fy_HV = HV**2 / H + 0.5 * G * H**2
    
    return (Fx_H, Fx_HU, Fx_HV), (Fy_H, Fy_HU, Fy_HV)

@jit
def update_step(state, _):
    """One timestep using a simple FDT (with basic artificial viscosity for stability)."""
    H, U, V = state
    
    # Compute fluxes
    (FxH, FxHU, FxHV), (FyH, FyHU, FyHV) = get_fluxes(H, U, V)
    
    # Simple central difference for spatial gradients
    def d_dx(arr):
        return (jnp.roll(arr, -1, axis=1) - jnp.roll(arr, 1, axis=1)) / (2 * DX)
    
    def d_dy(arr):
        return (jnp.roll(arr, -1, axis=0) - jnp.roll(arr, 1, axis=0)) / (2 * DY)

    # Evolution equations for conservative variables (H, HU, HV)
    HU = H * U
    HV = H * V
    
    # Time derivatives
    dH_dt = - (d_dx(FxH) + d_dy(FyH))
    dHU_dt = - (d_dx(FxHU) + d_dy(FyHU))
    dHV_dt = - (d_dx(FxHV) + d_dy(FyHV))
    
    # Artificial Viscosity (simple Laplacian smoothing) 
    # This prevents high-frequency oscillations (ringing) in simple FD schemes
    nu = 0.5
    def laplacian(arr):
        l = (jnp.roll(arr, -1, axis=1) + jnp.roll(arr, 1, axis=1) + 
             jnp.roll(arr, -1, axis=0) + jnp.roll(arr, 1, axis=0) - 4*arr)
        return l

    # Update conservative variables
    H_next = H + DT * dH_dt + nu * laplacian(H)
    HU_next = HU + DT * dHU_dt + nu * laplacian(HU)
    HV_next = HV + DT * dHV_dt + nu * laplacian(HV)
    
    # Primitive variables for next step
    U_next = HU_next / H_next
    V_next = HV_next / H_next
    
    return (H_next, U_next, V_next), H_next

def run_simulation():
    # Initial state
    H_init, U_init, V_init = initialize()
    
    # Use jax.lax.scan for efficient loop
    final_state, history = lax.scan(update_step, (H_init, U_init, V_init), None, length=STEPS)
    
    return history

if __name__ == "__main__":
    print("Simulating 2D Shallow Water Equations...")
    history = run_simulation()
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    times = [0, STEPS // 4, STEPS - 1]
    for i, t in enumerate(times):
        im = axes[i].imshow(history[t], origin='lower', cmap='Blues', extent=[0, L, 0, L])
        axes[i].set_title(f"Time Step: {t}")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.suptitle("Free Surface Elevation (H)", y=1.05, fontsize=16)
    plt.show()
    print("Done!")

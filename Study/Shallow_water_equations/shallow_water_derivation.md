# Derivation of Shallow Water Equations from Navier-Stokes

The Shallow Water Equations (SWE) are a set of hyperbolic partial differential equations that describe the flow below a pressure surface in a fluid. They are derived from the depth-integrated Navier-Stokes equations in the case where the horizontal length scale is much greater than the vertical length scale.

## 1. Starting Point: Navier-Stokes Equations

We start with the incompressible Navier-Stokes equations for a fluid with constant density $\rho$:

**Mass Conservation (Continuity):**
$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

**Momentum Conservation:**
$$\rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \rho \mathbf{g} + \mathbf{F}_{viscous}$$

where $\mathbf{u} = (u, v, w)$ is the velocity vector, $p$ is the pressure, and $\mathbf{g}$ is gravity (acting in the negative $z$ direction).

## 2. Key Assumptions

1.  **Shallow Water Assumption**: The horizontal scale $L$ is much larger than the depth $H$ ($H/L \ll 1$). This implies vertical accelerations are negligible, so $Dw/Dt \approx 0$.
2.  **Hydrostatic Balance**: From the vertical momentum equation, neglecting vertical acceleration and viscosity:
    $$\frac{\partial p}{\partial z} = -\rho g$$
    Integrating this from a depth $z$ to the free surface $\eta(x, y, t)$:
    $$p(z) = \rho g (\eta - z) + p_{atm}$$
    Assuming $p_{atm}$ is constant (or zero), the pressure is determined solely by the weight of the water column above.
3.  **Horizontal Velocity is Depth-Independent**: Because the fluid is shallow, we assume $u$ and $v$ do not vary significantly with $z$.

## 3. Depth Integration

We integrate the equations from the bottom $z = -h(x, y)$ to the surface $z = \eta(x, y, t)$. Let the total depth be $H = h + \eta$.

### Continuity Equation
Integrating $\int_{-h}^{\eta} (\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z}) dz = 0$:
Using Leibniz Integral Rule and the kinematic boundary conditions:
- At the surface ($z = \eta$): $w = \frac{\partial \eta}{\partial t} + u \frac{\partial \eta}{\partial x} + v \frac{\partial \eta}{\partial y}$
- At the bottom ($z = -h$): $w = -u \frac{\partial h}{\partial x} - v \frac{\partial h}{\partial y}$ (assuming rigid bottom)

We get the **Mass Conservation** part of SWE:
$$\frac{\partial H}{\partial t} + \frac{\partial (uH)}{\partial x} + \frac{\partial (vH)}{\partial y} = 0$$

### Momentum Equations
We use the hydrostatic pressure $p = \rho g (\eta - z)$. The horizontal pressure gradient is:
$$\frac{\partial p}{\partial x} = \rho g \frac{\partial \eta}{\partial x}$$
(Note this is independent of $z$).

Substituting into the horizontal momentum equations and integrating over depth:
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -g \frac{\partial \eta}{\partial x}$$
$$\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -g \frac{\partial \eta}{\partial y}$$

## 4. Final 2D Shallow Water Equations

In conservative form, including a source term for the bottom slope and neglecting friction:

1.  **Conservation of Mass:**
    $$\frac{\partial H}{\partial t} + \nabla \cdot (H\mathbf{u}) = 0$$
2.  **Conservation of Momentum:**
    $$\frac{\partial (Hu)}{\partial t} + \frac{\partial}{\partial x} (Hu^2 + \frac{1}{2}gH^2) + \frac{\partial}{\partial y} (Huv) = gH \frac{\partial h}{\partial x}$$
    $$\frac{\partial (Hv)}{\partial t} + \frac{\partial}{\partial x} (Huv) + \frac{\partial}{\partial y} (Hv^2 + \frac{1}{2}gH^2) = gH \frac{\partial h}{\partial y}$$

where:
- $H$ is total water depth ($\eta + h$).
- $u, v$ are depth-averaged horizontal velocities.
- $h$ is the bathymetry (depth of the floor from a reference level).
- $\eta$ is the free-surface elevation.

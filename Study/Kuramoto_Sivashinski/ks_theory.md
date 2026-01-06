# 1D Kuramoto Sivashinski (KS) System
$$\newcommand{\p}{\partial}$$ $$\frac{\p u}{\p t} = - \frac{1}{2} \frac{\p{u^2}}{\p x} - \frac{\partial^2 u}{\partial x^2} - \frac{\partial^4 u}{\partial x^4} $$

Unlike the Lorenz '63 system, the KS equation in $1D$ demonstrates chaos both in space and time, which makes it eligible to be tested on climate emulation techniques like `CorrDiff`. Recall that the solution to the lorenz system was described by $f(t)$ where the solution only depended on the time, $t$. However, in the 1D-KS equations, we describe solutions as a vector field $$u(x,t): \mathbb{R} \times \R \to \R$$ 

Let's briefly describe what the terms in the equation mean: 
- $\frac{\p u}{\p t}$ is the time evolution of the field. 
- $- \frac{1}{2} \frac{\p{u^2}}{\p x}$ describes the non-linear convection.
    - Recall that linear convection is described by $$\frac{\p u}{\p t} = -c \frac{\p u}{\p x} $$
    - It describes the propagation of an initial wave say $u(x,0) = u_0(x)$ at speed $c$ without change of shape. The exact solution to the system is give by: $$u(x,t) = u_0(x-ct) $$
    - To introduce non-linearity we replace $c$ with $u(x,t):$ $$-u\frac{\p u}{\p t}$$
    - This term is also famously referred to as the "Burgers" term which appears in the Inviscid Burgers' Equation: $$ \frac{\p u}{\p t} + u \frac{\p u}{\p x} = 0$$
    - Since $u$ is a scaler field (often referred to as the wave speed), the effect of introducing $u$ as a multiple in front of $\p u / \p x$ is that the variation wave speed in space depends on the scaler field or wave speed itself. 
- $- \frac{\partial^2 u}{\partial x^2}$ describes the negative diffusion
-  $-\frac{\partial^4 u}{\partial x^4}$ is the hyper diffusion term.
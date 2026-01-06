# Kuramoto Sivashinski Solver Using Exponential Time Differencing (ETD)

The simplest ETD method is also known as the exponential Euler method. The idea is that if the PDE in question has the following form: $$\newcommand{\p}{\partial} \newcommand{\l}{\mathcal{L}} \newcommand{\n}{\mathcal{N}} $$ $$ \frac{\p u}{\p t} = \l(u) + \n(u)$$
where $\l, ~\n$ are the linear and non-linear operators respectively, then we can express (and evolve) the state space system as: $$u_{[t+1]} = c~ u_{[t]} + \frac{c-1}{\l} \n(u_{[t]}) $$  where $c = e^{\l \Delta t}$. Before I move on to describe the algorithm, let me quickly discuss the main engine of this algorithm: the spectral method. 

## Fast Fourier Transform

Recall that Fourier transform allows us to represent a function in the Fourier basis. Discrete fourier transform expresses the process as a matrix multiplication: $$\bf \hat{X} = Wx$$ 
where $x$ is the vector sampled from the function (or scaler field in our case) in spatial domain, $\hat{X}$ is the representation of the wave in the frequency domain, and $W$ is the Discrete Fourier Transform (DFT) matrix. The DFT Matrix is given by: $$[W]_{k,\ell} = \exp(\frac{-2\pi i}{N}\cdot k \cdot \ell)$$ where $k$ is the frequency and $N$ is the size of the wave sample. DFT has $O(N^2)$ complexity, whereas FFT has $O(N\log N)$ complexity. FFT is simply an optimized version of the DST. In general, given the function $u(x,t)$ the input to DST (and FFT by extension) is a vector with values $u_t = u(0:{N-1}, t)$ where the first coordinate represents all $N$ grid points and $t$ represents time. So, in essence, the FFT operator takes one snapshot of the scaler field in space and transforms it to frequency domain.

### Why Frequency Domain

One of the advantages of using the frequency domain is that Spectral Derivative Method (SDM) is much more accurate than Finite Difference Method (FDM). The FDM is usually given by the central difference formula: $$u_t'(x_i) \approx \frac{u_t(x_{i+1}) - u_t(x_{i-1})}{2\Delta x}$$ Besides the precision problem, there are issues with stability of numerical solutions such as Numerical diffusion and unwind-ing problem. SDM, by comparison, does not suffer from any directional bias (seen in the FDM) or numerical diffusion. In the frequency domain, the derivative operator is given by: $$\frac{d}{dx} \to (ik) $$ where $k$ is the wave number. 

## The Algorithm

With the theory out of the way, we can focus on the algorithm itself. Just as a reminder, the KS equation is given by:  $$\frac{\p u}{\p t} = - \frac{1}{2} \frac{\p{u^2}}{\p x} - \frac{\partial^2 u}{\partial x^2} - \frac{\partial^4 u}{\partial x^4} $$


- Compute the linear derivative of the KS equation in the frequency domain: $$ \hat \l = -\hat{D}^2 - \hat{D}^4 = -(ik)^2 - (ik)^4  = k^2 - k^4$$
- Compute the constant: $$c =  e^{ \hat \l \Delta t} = e^{(k^2 - k^4)\Delta t}$$
- Note that if $\hat \l  = 0$ (happens when $k=0$), then set $$ \frac{e^{\hat \l \Delta t} -1}{\hat \l} \to \Delta t$$

The algorithm is the following:

- 
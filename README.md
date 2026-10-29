
# Dupire Local Volatility Model

**Python Implementation | Volatility Surface Calibration | Monte Carlo Repricing**

## 1. Overview

This project implements the **Dupire Local Volatility Model** in Python, from the construction of a synthetic implied volatility surface to local volatility calibration and European option repricing.

The objective is to understand how market-implied volatility information can be transformed into a state- and time-dependent volatility function, and to evaluate whether the resulting model can reproduce European call option prices.

The project also examines the numerical challenges associated with Dupire calibration, particularly the instability of price derivatives at short maturities.

## 2. Mathematical Framework

### 2.1. Local Volatility Dynamics

Under the risk-neutral measure, the underlying asset follows:

```math
dS_t = (r-q)S_t\,dt + \sigma_{\mathrm{loc}}(S_t,t)S_t\,dW_t
```

where:

- $S_t$ is the underlying asset price.
- $r$ is the risk-free interest rate.
- $q$ is the continuous dividend yield.
- $\sigma_{\mathrm{loc}}(S_t,t)$ is the instantaneous local volatility.
- $W_t$ is a Brownian motion under the risk-neutral measure.

Unlike the Black-Scholes model, which assumes constant volatility, the local volatility framework allows volatility to depend on both the underlying asset price and time.

### 2.2. Dupire's Formula

Dupire's formula recovers local variance from a surface of European call option prices:

```math
\sigma_{\mathrm{loc}}^2(K,T)
=
\frac{
\frac{\partial C}{\partial T}
+
(r-q)K\frac{\partial C}{\partial K}
+
qC
}{
\frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}
}
```

Here, $C(K,T)$ denotes the European call option price for strike $K$ and maturity $T$.

The formula requires three derivatives.

**Maturity derivative**

```math
\frac{\partial C}{\partial T}
```

Measures how the call price changes as maturity increases, holding the strike constant.

**First strike derivative**

```math
\frac{\partial C}{\partial K}
```

Measures the sensitivity of the call price to changes in strike.

**Second strike derivative**

```math
\frac{\partial^2 C}{\partial K^2}
```

Measures the curvature of the call price with respect to strike.

Under standard no-arbitrage assumptions, European call prices must be convex in strike. The second derivative is therefore non-negative and is related to the risk-neutral probability density of the underlying asset at maturity.

### 2.3. Numerical Differentiation

The initial implementation used finite differences to approximate the derivatives required by Dupire's formula.

For a strike increment $h$, the first strike derivative can be approximated by:

```math
\frac{\partial C}{\partial K}
\approx
\frac{C(K+h,T)-C(K-h,T)}{2h}
```

The second strike derivative is approximated by:

```math
\frac{\partial^2 C}{\partial K^2}
\approx
\frac{C(K+h,T)-2C(K,T)+C(K-h,T)}{h^2}
```

Similarly, the maturity derivative can be approximated by:

```math
\frac{\partial C}{\partial T}
\approx
\frac{C(K,T+\Delta T)-C(K,T-\Delta T)}{2\Delta T}
```

Although straightforward, finite differences can introduce numerical errors, particularly at short maturities and for deep in-the-money or out-of-the-money options.

In these regions, the second strike derivative may become extremely small. Since it appears in the denominator of Dupire's formula, numerical inaccuracies can generate unrealistically large or undefined local volatility estimates.

During development, the initial numerical implementation produced local volatility spikes approaching 1,000% and invalid values that propagated into the Monte Carlo simulation.

### 2.4. Analytical Derivatives

To improve numerical stability, the final implementation uses an analytical formulation based on the derivatives of the synthetic implied volatility surface.

The implied volatility function is:

```math
\sigma_{\mathrm{imp}}(K,T)
=
0.20
+
0.10\left(\frac{K}{100}-1\right)^2
+
0.02T
```

This function incorporates a quadratic volatility smile and an upward-sloping term structure.

Its first derivative with respect to strike is:

```math
\frac{\partial \sigma_{\mathrm{imp}}}{\partial K}
=
\frac{0.20}{100}
\left(\frac{K}{100}-1\right)
```

Its second derivative with respect to strike is:

```math
\frac{\partial^2 \sigma_{\mathrm{imp}}}{\partial K^2}
=
\frac{0.20}{100^2}
```

The maturity derivative is:

```math
\frac{\partial \sigma_{\mathrm{imp}}}{\partial T}
=
0.02
```

These derivatives are incorporated into an analytical reformulation of Dupire's equation based on Black-Scholes implied volatility.

This avoids estimating the second strike derivative of option prices through successive finite differences and improves numerical stability for the synthetic surface.

## 3. Python Implementation

The project follows nine stages:

| Stage | Description |
|---|---|
| 1. Market Data | Construction of a synthetic strike and maturity grid |
| 2. Black-Scholes | European call option pricing |
| 3. Implied Volatility | Generation of a synthetic volatility surface |
| 4. Interpolation | Cubic spline interpolation |
| 5. Option Prices | Construction of the call price surface |
| 6. Derivatives | Analytical implied volatility derivatives |
| 7. Dupire Calibration | Reconstruction of local volatility |
| 8. Surface Comparison | 3D visualization of implied and local volatility |
| 9. Repricing | Monte Carlo pricing validation |

## 4. Implied Volatility vs Local Volatility

Implied volatility and local volatility represent different quantities.

**Implied volatility** is the constant volatility parameter that reproduces an observed option price under the Black-Scholes framework.

**Local volatility** is the instantaneous volatility assigned by the model to a specific underlying asset price and time.

Consequently, the two volatility surfaces are not expected to be identical.

The calibration objective is to construct a local volatility function capable of reproducing the European call option price surface, rather than matching implied volatility point by point.

## 5. Monte Carlo Repricing

After calibration, the underlying asset is simulated under the risk-neutral measure using a log-Euler discretization:

```math
S_{t+\Delta t}
=
S_t
\exp\left[
\left(r-q-\frac{1}{2}\sigma_{\mathrm{loc}}^2(S_t,t)\right)\Delta t
+
\sigma_{\mathrm{loc}}(S_t,t)\sqrt{\Delta t}\,Z
\right]
```

where $Z$ is a standard normal random variable.

European call prices are estimated from discounted simulated payoffs:

```math
C_{\mathrm{model}}(K,T)
=
e^{-rT}
\frac{1}{N}
\sum_{i=1}^{N}
\max(S_T^{(i)}-K,0)
```

Pricing accuracy is measured using the root mean squared error:

```math
\mathrm{RMSE}
=
\sqrt{
\frac{1}{M}
\sum_{j=1}^{M}
\left(
C_{\mathrm{model},j}
-
C_{\mathrm{market},j}
\right)^2
}
```

## 6. Model Configuration

| Parameter | Value |
|---|---|
| Initial underlying price | 100 |
| Risk-free interest rate | 2% |
| Dividend yield | 0% |
| Strike range | 60–140 |
| Maturity range | 0.01–2 years |
| Monte Carlo paths | 200,000 |
| Simulation time steps | 800 |
| Validation options | 25 European calls |

The validation options cover strikes of 80, 90, 100, 110 and 120, with maturities of 0.25, 0.50, 1.00, 1.50 and 2.00 years.

## 7. Calibration Results

The final implementation achieved the following results:

| Metric | Result |
|---|---|
| Validation options | 25 |
| Monte Carlo paths | 200,000 |
| Time steps | 800 |
| Final pricing RMSE | **0.044926** |
| Maximum absolute pricing error | 0.1373 |
| Previous pricing RMSE | 0.097266 |

The final observed RMSE represents an approximately **54% reduction** relative to the earlier numerical implementation.

The analytical formulation improved the numerical stability of the local volatility surface and produced relatively small pricing errors for the selected synthetic European calls.

The reported results depend on the synthetic volatility surface, simulation seed, and numerical settings. The comparison does not independently isolate Monte Carlo sampling effects.

## 8. Visualizations

The implementation produces three-dimensional plots of:

1. The interpolated implied volatility surface.
2. The calibrated Dupire local volatility surface.
3. The comparison between implied and local volatility.

These visualizations illustrate how local volatility differs from implied volatility while being constructed to reproduce the same European option price surface.

## 9. Installation and Usage

Clone the repository:

```bash
git clone https://github.com/Yasslbi/dupire-local-volatility.git
cd dupire-local-volatility
```

Install the required dependencies:

```bash
pip install numpy pandas scipy matplotlib
```

Run the model:

```bash
python dupire_local_volatility.py
```

The program calculates the volatility surfaces, performs Monte Carlo repricing, displays the pricing comparison table, and reports the RMSE.

## 10. Limitations

This implementation is intended for educational purposes rather than production use.

The main limitations include:

- Synthetic implied volatility data rather than observed market quotes.
- Constant interest rates and dividend yields.
- Finite strike and maturity domains requiring extrapolation.
- Monte Carlo statistical uncertainty and time-discretization errors.
- No comprehensive static-arbitrage cleaning procedure.
- No formal calibration tolerance or production-level validation framework.

## 11. Future Improvements

Potential extensions include:

- Calibration to real market option data.
- Arbitrage-free implied volatility surface construction.
- Monte Carlo variance reduction techniques.
- Pricing error analysis in implied volatility points.
- Comparison with stochastic volatility models such as Heston.
- Extension to exotic derivatives pricing.

## 12. References

- Dupire, B. (1994). *Pricing with a Smile*. Risk.
- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.

---

**Disclaimer:** This project is intended for educational and quantitative finance research purposes only. It is not a production-ready pricing or risk management framework.




import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# EXERCISE 1 - MARKET DATA

S0 = 100.0
r = 0.02
q = 0.0

strikes = np.arange(60, 141, 5)

maturities = np.array([
    0.01, 0.05, 0.10, 0.25,
    0.50, 1.00, 1.50, 2.00
])

df = pd.MultiIndex.from_product(
    [strikes, maturities],
    names=["Strike", "Maturity"]
).to_frame(index=False)


# EXERCISE 2 - BLACK-SCHOLES CALL PRICING

def black_scholes_call(S0, K, T, r, q, sigma):

    d1 = (
        np.log(S0 / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    call_price = (
        S0 * np.exp(-q * T) * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )

    return call_price


# EXERCISE 3 - IMPLIED VOLATILITY SURFACE

def implied_vol(K, T):

    moneyness = K / S0 - 1

    vol = 0.20 + 0.10 * moneyness**2 + 0.02 * T

    return vol


df["Implied_Vol"] = implied_vol(
    df["Strike"],
    df["Maturity"]
)


# EXERCISE 4 - IMPLIED VOLATILITY INTERPOLATION

# 1. Create volatility matrix

vol_matrix = df.pivot(
    index="Maturity",
    columns="Strike",
    values="Implied_Vol"
)

print("\nIMPLIED VOLATILITY TABLE (%)")
print((vol_matrix * 100).round(2))


# 2. Cubic spline interpolation

interpolator = RectBivariateSpline(
    maturities,
    strikes,
    vol_matrix.values,
    kx=3,
    ky=3
)


# 3. Create finer grid

strikes_fine = np.linspace(60, 140, 161)
maturities_fine = np.linspace(0.01, 2.0, 200)

K, T = np.meshgrid(
    strikes_fine,
    maturities_fine
)


# 4. Interpolate implied volatility

vol_fine = interpolator(
    maturities_fine,
    strikes_fine
)


# EXERCISE 5 - MARKET CALL PRICES

# 1. Market prices

df["Call_Price"] = black_scholes_call(
    S0,
    df["Strike"],
    df["Maturity"],
    r,
    q,
    df["Implied_Vol"]
)


# 2. Fine-grid call prices

call_prices_fine = black_scholes_call(
    S0, K, T, r, q, vol_fine
)


# EXERCISE 6 - ANALYTICAL DERIVATIVES

# 1. Implied volatility

sigma = implied_vol(K, T)


# 2. First derivative with respect to strike

sigma_K = (
    0.20 * (K / S0 - 1) / S0
)


# 3. Second derivative with respect to strike

sigma_KK = np.full_like(
    K,
    0.20 / S0**2
)


# 4. First derivative with respect to maturity

sigma_T = np.full_like(
    K,
    0.02
)


# 5. Black-Scholes d1 and d2

sqrt_T = np.sqrt(T)

d1 = (
    np.log(S0 / K)
    + (r - q + 0.5 * sigma**2) * T
) / (sigma * sqrt_T)

d2 = d1 - sigma * sqrt_T


# EXERCISE 7 - LOCAL VOLATILITY CALIBRATION

# 1. Stable Dupire numerator

scaled_numerator = (
    sigma / (2 * T)
    + sigma_T
    + (r - q) * K * sigma_K
)


# 2. Stable Dupire denominator

scaled_denominator = 0.5 * (
    1 / (sigma * T)
    + 2 * K * d1 * sigma_K / (sigma * sqrt_T)
    + K**2 * d1 * d2 * sigma_K**2 / sigma
    + K**2 * sigma_KK
)


# 3. Check denominator

if np.any(~np.isfinite(scaled_denominator)) or np.any(scaled_denominator <= 0):
    raise ValueError(
        "Invalid Dupire denominator: check the implied volatility surface."
    )


# 4. Calculate local variance

local_variance = (
    scaled_numerator / scaled_denominator
)

if np.any(~np.isfinite(local_variance)) or np.any(local_variance <= 0):
    raise ValueError(
        "Invalid local variance: check the implied volatility surface."
    )


# 5. Calculate local volatility

local_vol = np.sqrt(local_variance)


# 6. Calibration checks

print("\nLOCAL VOLATILITY CALIBRATION")

print(f"Minimum Local Vol: {np.min(local_vol)*100:.2f}%")
print(f"Maximum Local Vol: {np.max(local_vol)*100:.2f}%")
print(f"Invalid Points: {np.sum(~np.isfinite(local_vol))}")


# EXERCISE 8 - IMPLIED VS LOCAL VOLATILITY

# 1. Calculate volatility difference

vol_difference = (
    local_vol - vol_fine
) * 100


# 2. Comparison table

vol_comparison = pd.DataFrame({
    "Strike": K.ravel(),
    "Maturity": T.ravel(),
    "Implied_Vol": (vol_fine * 100).ravel(),
    "Local_Vol": (local_vol * 100).ravel(),
    "Difference": vol_difference.ravel()
})

print("\nVOLATILITY COMPARISON - SAMPLE")

print(
    vol_comparison.iloc[::4000].round(2)
)


# 3. Plot both surfaces

fig = plt.figure(figsize=(11, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot_surface(
    K,
    T,
    vol_fine * 100,
    color="royalblue",
    alpha=0.55,
    edgecolor="none"
)

ax.plot_surface(
    K,
    T,
    local_vol * 100,
    color="orange",
    alpha=0.55,
    edgecolor="none"
)

ax.set_xlabel("Strike")
ax.set_ylabel("Maturity (Years)")
ax.set_zlabel("Volatility (%)")

ax.set_title(
    "Implied Volatility vs Local Volatility"
)

ax.legend(handles=[
    Patch(
        facecolor="royalblue",
        label="Implied Volatility"
    ),
    Patch(
        facecolor="orange",
        label="Local Volatility"
    )
])

plt.show()


# EXERCISE 9 - LOCAL VOLATILITY REPRICING

# 1. Local volatility interpolator

local_vol_interpolator = RegularGridInterpolator(
    (maturities_fine, strikes_fine),
    local_vol,
    bounds_error=False,
    fill_value=None
)


# 2. Monte Carlo parameters

n_paths = 200000
n_steps = 800

dt = 2.0 / n_steps

rng = np.random.default_rng(42)

S = np.full(
    n_paths,
    S0,
    dtype=float
)


# 3. Validation instruments

validation_maturities = np.array([
    0.25, 0.50, 1.00, 1.50, 2.00
])

validation_strikes = np.array([
    80, 90, 100, 110, 120
])

step_counts = np.rint(
    validation_maturities / dt
).astype(int)


# 4. Simulate underlying prices

results = []

for step in range(1, n_steps + 1):

    t = (step - 1) * dt

    t_eval = np.clip(
        t,
        maturities_fine[0],
        maturities_fine[-1]
    )

    S_eval = np.clip(
        S,
        strikes_fine[0],
        strikes_fine[-1]
    )

    points = np.column_stack((
        np.full(n_paths, t_eval),
        S_eval
    ))

    sigma_local = local_vol_interpolator(
        points
    )

    # Safety check

    if not np.all(np.isfinite(sigma_local)):
        raise ValueError(
            f"Invalid local volatility at step {step}"
        )

    Z = rng.standard_normal(n_paths)

    # Log-Euler simulation

    S *= np.exp(
        (r - q - 0.5 * sigma_local**2) * dt
        + sigma_local * np.sqrt(dt) * Z
    )

    # Price calls at each maturity

    if step in step_counts:

        T_current = step * dt

        for strike in validation_strikes:

            payoff = np.maximum(
                S - strike,
                0
            )

            model_price = (
                np.exp(-r * T_current)
                * np.mean(payoff)
            )

            results.append({
                "Strike": strike,
                "Maturity": T_current,
                "Model_Price": model_price
            })


# 5. Compare market and model prices

model_df = pd.DataFrame(results)

market_df = df[
    df["Strike"].isin(validation_strikes)
    & df["Maturity"].isin(validation_maturities)
]

comparison = market_df.merge(
    model_df,
    on=["Strike", "Maturity"],
    validate="one_to_one"
)

if len(comparison) != 25:
    raise ValueError(
        f"Expected 25 options, got {len(comparison)}"
    )


# 6. Calculate pricing errors

comparison["Pricing_Error"] = (
    comparison["Model_Price"]
    - comparison["Call_Price"]
)


# 7. Calculate RMSE

rmse = np.sqrt(
    np.mean(
        comparison["Pricing_Error"]**2
    )
)


# 8. Display results

print("\nLOCAL VOLATILITY - PRICING VALIDATION")

print(
    comparison[
        [
            "Strike",
            "Maturity",
            "Call_Price",
            "Model_Price",
            "Pricing_Error"
        ]
    ].round(4).to_string(index=False)
)

print(f"\nRMSE: {rmse:.6f}")


# PLOT 1 - IMPLIED VOLATILITY SURFACE

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    K, T, vol_fine * 100,
    cmap="viridis",
    edgecolor="none"
)

ax.set_xlabel("Strike")
ax.set_ylabel("Maturity (Years)")
ax.set_zlabel("Implied Volatility (%)")
ax.set_title("Implied Volatility Surface")

fig.colorbar(surf, ax=ax, shrink=0.5)
plt.show()


# PLOT 2 - LOCAL VOLATILITY SURFACE

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    K, T, local_vol * 100,
    cmap="viridis",
    edgecolor="none"
)

ax.set_xlabel("Strike")
ax.set_ylabel("Maturity (Years)")
ax.set_zlabel("Local Volatility (%)")
ax.set_title("Dupire Local Volatility Surface")

fig.colorbar(surf, ax=ax, shrink=0.5)
plt.show()


import numpy as np

def run_monte_carlo_extrapolation(data_dict, N_sim, verbose=True):
    """
    Génère des régressions linéaires synthétiques via Monte Carlo.
    """
    X_means, Y_means = data_dict["X_means"], data_dict["Y_means"]
    X_stds, Y_stds = data_dict["X_stds"], data_dict["Y_stds"]
    exact_gs = data_dict["exact_gs_energy_per_site"]
    M = len(X_means)

    if verbose:
        print(f"Generating {N_sim} synthetic linear regressions...")

    X_sim = np.random.normal(loc=X_means[:, None], scale=X_stds[:, None], size=(M, N_sim))
    Y_sim = np.random.normal(loc=Y_means[:, None], scale=Y_stds[:, None], size=(M, N_sim))

    sum_X  = np.sum(X_sim, axis=0)
    sum_Y  = np.sum(Y_sim, axis=0)
    sum_X2 = np.sum(X_sim**2, axis=0)
    sum_XY = np.sum(X_sim * Y_sim, axis=0)

    Delta = M * sum_X2 - sum_X**2
    b_sim = (sum_X2 * sum_Y - sum_X * sum_XY) / Delta  
    a_sim = (M * sum_XY - sum_X * sum_Y) / Delta    

    E_extrap_mean = np.mean(b_sim)
    E_extrap_std = np.std(b_sim)
    ci_lower = np.percentile(b_sim, 2.5)
    ci_upper = np.percentile(b_sim, 97.5)

    if verbose:
        print(f"\n========== GLOBAL MONTE CARLO RESULTS ==========")
        print(f"Points used for fit       : {M} ({M//2} simulations)")
        print(f"Exact Energy              : {exact_gs:.8f}")
        print(f"Extrapolated Energy       : {E_extrap_mean:.8f} ± {E_extrap_std:.8f}")
        print(f"95% Confidence Interval   : [{ci_lower:.8f} , {ci_upper:.8f}]")
        print(f"Absolute Error vs Exact   : {abs(E_extrap_mean - exact_gs):.2e}")
        print(f"==================================================")

    return {
        "b_sim": b_sim,
        "a_sim": a_sim,
        "E_extrap_mean": E_extrap_mean,
        "E_extrap_std": E_extrap_std
    }
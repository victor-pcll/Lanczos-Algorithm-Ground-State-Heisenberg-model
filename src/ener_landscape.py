from tqdm import tqdm 
import numpy as np
from .utils import moyenne_glissante_convolve

def compute_lanczos_landscape(vs_init, ha, h2, alphas_to_test, n_mcmc_steps=500):
    """
    Génère le paysage d'énergie de Lanczos par reweighting stochastique.
    """
    all_curves = [] 

    for _ in tqdm(range(n_mcmc_steps), desc="MCMC Sampling & Reweighting"):
        vs_init.sample()
        
        eloc_np = np.array(vs_init.local_estimators(ha)).real
        h2loc_np = np.array(vs_init.local_estimators(h2)).real
        
        e_curve = []
        for a in alphas_to_test:
            weights = (1.0 + a * eloc_np)**2
            E_loc_1 = (eloc_np + a * h2loc_np) / (1.0 + a * eloc_np)
            e_approx = np.sum(weights * E_loc_1) / np.sum(weights)
            e_curve.append(e_approx)
            
        all_curves.append(e_curve)

    all_curves = np.array(all_curves)

    energies_mean = np.mean(all_curves, axis=0)
    energies_std = np.std(all_curves, axis=0, ddof=1) 
    e_vmc = energies_mean[0]
    
    return energies_mean, energies_std, e_vmc

def extract_optimal_alpha_bootstrap(alphas_to_test, energies_mean, energies_std, 
                                    n_mcmc_steps, n_bootstrap=10000, window_size=None):
    """
    Extrait le minimum et estime l'erreur via MA-Bootstrap.
    Le lissage (moyenne glissante) est optionnel.
    """
    energies_sem = energies_std / np.sqrt(n_mcmc_steps)
    
    if window_size is not None and window_size > 1:
        offset = (window_size - 1) // 2
        alphas_valides = alphas_to_test[offset : offset + (len(alphas_to_test) - window_size + 1)]
        energies_valides = moyenne_glissante_convolve(energies_mean, window_size)
    else:
        alphas_valides = alphas_to_test
        energies_valides = energies_mean

    idx_min_reel = np.argmin(energies_valides)
    alpha_opt_final = alphas_valides[idx_min_reel]
    e_min_final = energies_valides[idx_min_reel]

    alphas_min_bootstrap = []
    print(f"\nRunning Bootstrap for Error Estimation (Smoothing: {window_size is not None})...")
    
    for _ in range(n_bootstrap):
        y_virtual = np.random.normal(energies_mean, energies_sem) # Le Bootstrap Paramétrique, on genre des nouvelles simulations d'énergie à partir de la distribution normale centrée sur les énergies moyennes et avec l'écart-type des SEM.
        # on fait une approx guassienne de la distribution des énergies à chaque alpha, et on génère une nouvelle "courbe virtuelle" d'énergie à partir de cette distribution.
        if window_size is not None and window_size > 1:
            y_virtual_valide = moyenne_glissante_convolve(y_virtual, window_size)
        else:
            y_virtual_valide = y_virtual
            
        idx_min_v = np.argmin(y_virtual_valide)
        alphas_min_bootstrap.append(alphas_valides[idx_min_v])

    alpha_opt_err = np.std(alphas_min_bootstrap, ddof=1)
    
    return {
        "alpha_opt_final": alpha_opt_final,
        "e_min_final": e_min_final,
        "alpha_opt_err": alpha_opt_err,
        "alphas_min_bootstrap": alphas_min_bootstrap
    }
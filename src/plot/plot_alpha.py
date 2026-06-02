import matplotlib.pyplot as plt
import os
import numpy as np
from .plot_params import *

def plot_lanczos_results(alphas_to_test, energies_mean, energies_std, e_vmc, exact_gs_energy, 
                         bootstrap_results, save_dir=save_default):
    """
    Génère et sauvegarde les graphiques de paysage d'énergie et de distribution bootstrap.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extraction des résultats
    alpha_opt_final = bootstrap_results["alpha_opt_final"]
    e_min_final = bootstrap_results["e_min_final"]
    alpha_opt_err = bootstrap_results["alpha_opt_err"]
    alphas_min_bootstrap = bootstrap_results["alphas_min_bootstrap"]

    # --- GRAPHIC 1 : Le paysage d'énergie lissé ---
    plt.figure(figsize=fig_size, dpi=120)

    # 1. Courbe et erreur
    plt.plot(alphas_to_test, energies_mean, color="red", label=r"Landscape $E_L(\alpha)$")
    plt.fill_between(alphas_to_test, energies_mean - energies_std, energies_mean + energies_std, 
                     color="#e63946", alpha=0.15, label=r"Uncertainty ($\pm 1 \sigma_{\mathrm{SEM}}$)")

    # 2. L'optimum (SANS les résultats numériques)
    plt.axvline(x=alpha_opt_final, color="#1d3557", linestyle="--", linewidth=2, 
                label=r"Extracted optimum $\alpha_{\mathrm{opt}}$")
    plt.scatter([alpha_opt_final], [e_min_final], color="#1d3557", s=100, marker='o', zorder=5)
    plt.axvspan(alpha_opt_final - alpha_opt_err, alpha_opt_final + alpha_opt_err, color="#1d3557", alpha=0.25)

    # 3. Énergies de référence (SANS les résultats numériques)
    plt.axhline(y=e_vmc, color="#457b9d", linestyle=":", linewidth=2, 
                label=r"Baseline $E_{\mathrm{VMC}}$")
    plt.axhline(y=exact_gs_energy, color="#2a9d8f", linestyle="-.", linewidth=2.5, alpha=0.9, 
                label=r"Exact $E_{\mathrm{ex}}$")

    # Axes et affichage
    plt.xlabel(r"Lanczos Parameter $\alpha$", fontsize=LABEL_FONTSIZE)
    plt.ylabel(r"Lanczos Energy $E_L(\alpha)$", fontsize=LABEL_FONTSIZE)
    plt.ylim(exact_gs_energy - 0.005, e_vmc + 0.01) 
    
    # La légende prendra beaucoup moins de place maintenant
    plt.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.9, edgecolor="black")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lanczos_landscape_bootstrap_MA.png"), dpi=300)
    plt.show()

    # --- GRAPHIC 2 : Distribution de l'incertitude ---
    plt.figure(figsize=fig_size, dpi=120)
    plt.hist(alphas_min_bootstrap, bins=max(10, len(np.unique(alphas_min_bootstrap))), density=True, color="#457b9d", alpha=0.75, edgecolor="white", label="Bootstrap samples")

    plt.axvline(x=alpha_opt_final, color="#e63946", linestyle="-", linewidth=2.5, label=f"Selected $\\alpha$ = {alpha_opt_final:.4f}")
    plt.axvline(x=np.mean(alphas_min_bootstrap), color="#2a9d8f", linestyle="-.", linewidth=2, label=f"Bootstrap Mean $\\alpha$ = {np.mean(alphas_min_bootstrap):.4f}")
    plt.axvspan(alpha_opt_final - alpha_opt_err, alpha_opt_final + alpha_opt_err, color="#1d3557", alpha=0.25)

    plt.xlabel(r"Optimal Parameter Value $\alpha$", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Probability Density", fontsize=LABEL_FONTSIZE)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.9, edgecolor="black")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lanczos_alpha_distribution_MA.png"), dpi=300)
    plt.show()

    #--- GRPAHIC 3 : Error ---
    plt.figure(figsize=fig_size, dpi=120)

    plt.plot(alphas_to_test, np.abs(energies_mean - exact_gs_energy),  label="Absolute error")
    plt.axvline(x=alpha_opt_final, color="#1d3557", linestyle="--", linewidth=2, 
                label=f"$\\alpha_{{opt}}$ = {alpha_opt_final:.4f} $\\pm$ {alpha_opt_err:.4f}")
    plt.axvspan(alpha_opt_final - alpha_opt_err, alpha_opt_final + alpha_opt_err, color="#1d3557", alpha=0.25)

    plt.xlabel(r"Optimal Parameter Value $\alpha$", fontsize=LABEL_FONTSIZE)
    plt.ylabel(r"$\Delta E$", fontsize=LABEL_FONTSIZE)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.yscale("log")
    plt.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.9, edgecolor="black")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lanczos_alpha_abs_error.png"), dpi=300)
    plt.show()

    # --- Bilan final ---
    gain = e_vmc - e_min_final
    print("\n" + "="*50)
    print(f"➔ EXACT MA ALPHA  : {alpha_opt_final:.5f} ± {alpha_opt_err:.5f}")
    print(f"➔ ENERGY GAIN     : {gain:.5f} (VMC: {e_vmc:.4f} ➔ Lanczos: {e_min_final:.4f})")
    print("="*50)

def plot_alpha_convergence(historique_alpha, alpha_min, alpha_err, 
                           data_file="../../assets/", tail_length=100):

    os.makedirs(data_file, exist_ok=True)
    historique_alpha = np.array(historique_alpha)
    
    # Statistiques sur la queue de la chaîne (la zone convergée)
    tail = min(tail_length, len(historique_alpha)) 
    tail_alphas = historique_alpha[-tail:]
    alpha_mean = np.mean(tail_alphas)
    alpha_std = np.std(tail_alphas, ddof=1)

    print(f" -> Target Bootstrap Alpha : {alpha_min:.6f} ± {alpha_err:.6f}")
    print(f" -> Optimized VMC Alpha    : {alpha_mean:.6f} ± {alpha_std:.6f} (last {tail} iters)")

    # ==========================================
    # FIGURE 1 : Trajectoire de Alpha
    # ==========================================
    plt.figure(figsize=fig_size, dpi=120)
    
    # 1. Trajectoire de l'optimisation (Au lieu de "VMC \alpha")
    plt.plot(historique_alpha, marker='o', markersize=4, color="#457b9d", linewidth=1.5, alpha=0.7, label=r"SGD Trajectory")
    
    # 2. Cible et Erreur Bootstrap (Au lieu de "Target \alpha_{opt}")
    # On utilise \mathrm{opt} pour respecter la typographie LaTeX du texte
    plt.axhline(y=alpha_min, color="#2a9d8f", linestyle="--", linewidth=2.5, label=r"$\alpha_{\mathrm{opt}}$ (MA-Bootstrap)")
    plt.axhspan(alpha_min - alpha_err, alpha_min + alpha_err, color="#2a9d8f", alpha=0.25)

    # 3. Moyenne et Bruit (Au lieu de "VMC Mean")
    plt.axhline(y=alpha_mean, color="gray", linestyle=":", linewidth=2.5, label=r"Steady-State Mean")
    plt.axhspan(alpha_mean - alpha_std, alpha_mean + alpha_std, color='gray', alpha=0.15)

    # L'axe X : le texte dit "over 1000 VMC iterations" ou "SGD iterations"
    plt.xlabel("Optimization Iteration", fontsize=LABEL_FONTSIZE) 
    plt.ylabel(r"Lanczos Parameter $\alpha$", fontsize=LABEL_FONTSIZE)
    plt.grid(True, linestyle="--")
    plt.legend(loc="best", framealpha=0.9, edgecolor="black", fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_file, "alpha_trajectory.png"), bbox_inches='tight')
    plt.show()

    # ==========================================
    # FIGURE 2 : Erreur Absolue (Échelle Log)
    # ==========================================
    plt.figure(figsize=fig_size, dpi=150)
    
    epsilon = 1e-12 
    diff_alpha = np.abs(historique_alpha - alpha_min) + epsilon

    plt.plot(diff_alpha, marker='o', markersize=4, color="#1d3557", linewidth=1.5, alpha=0.7, label=r"$|\alpha - \alpha_{opt}|$")
    plt.axhline(y=alpha_err, color="#2a9d8f", linestyle="--", linewidth=2.5, label="Uncertainty Floor")

    plt.yscale("log")
    plt.xlabel("VMC Iteration", fontsize=LABEL_FONTSIZE)
    plt.ylabel(r"Absolute Error (log scale)", fontsize=LABEL_FONTSIZE)
    plt.grid(True, which="both", linestyle="--") # Active la grille mineure pour le log
    plt.legend(loc="best", framealpha=0.9, edgecolor="black", fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_file, "alpha_error_log.png"), bbox_inches='tight')
    plt.show()

    # ==========================================
    # FIGURE 3 : Histogramme de la zone convergée
    # ==========================================
    plt.figure(figsize=fig_size, dpi=150)
    
    # Histogramme des dernières itérations
    plt.hist(tail_alphas, bins=15, color="#fa5703", alpha=0.75, edgecolor="white", density=True, label="VMC Samples")
    
    plt.axvline(x=alpha_mean, color="#1d3557", linestyle="-", linewidth=2.5, label=r"Mean $\alpha$")
    plt.axvspan(alpha_mean - alpha_std, alpha_mean + alpha_std, color="#1d3557", alpha=0.2, label=r"VMC Noise ($\pm 1 \sigma$)")

    plt.axvspan(alpha_min - alpha_err, alpha_min + alpha_err, color="#2a9d8f", alpha=0.2, label=r"$\alpha_{opt}$ error")
    
    # Optionnel : Ajouter la cible Bootstrap pour comparer visuellement la distribution à l'objectif
    plt.axvline(x=alpha_min, color="#2a9d8f", linestyle="--", linewidth=2.5, label=r"Target $\alpha_{opt}$")

    plt.xlabel(r"Sampled $\alpha$ values", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Probability Density", fontsize=LABEL_FONTSIZE)
    plt.grid(True, linestyle="--")
    plt.legend(loc="best", framealpha=0.9, edgecolor="black", fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_file, "alpha_histogram.png"), bbox_inches='tight')
    plt.show()

    # Nettoyage des paramètres pour ne pas affecter la suite de ton code
    plt.rcdefaults()
    
    return alpha_mean, alpha_std
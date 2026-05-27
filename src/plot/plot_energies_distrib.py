import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
import os
import numpy as np
import matplotlib.pyplot as plt
from .plot_params import *

def plot_energy_distribution(el_valides, results_e0, exact_gs_energy, data_file, bins=40, save_file=False):
    """
    Génère l'histogramme des distributions d'énergie (VMC vs Lanczos).
    Optimisé pour un rendu LaTeX/Overleaf de haute qualité.
    """
    kde = stats.gaussian_kde(el_valides)
    x_grid = np.linspace(np.min(el_valides), np.max(el_valides), 1000) 
    kde_curve = kde(x_grid)
    mode_EL = x_grid[np.argmax(kde_curve)]

    plt.figure(figsize=fig_size, dpi=120)

    # 3. Tracé des histogrammes (Hiérarchie visuelle via l'alpha et le zorder)
    # L'état initial VMC a une plus grande variance. On le met derrière (zorder=1) 
    # et plus transparent (alpha=0.4) pour ne pas masquer Lanczos.
    plt.hist(results_e0, bins=bins, color="#f4a261", edgecolor="black", 
             alpha=0.4, density=True, label=r"Dist. $E_0$ (VMC)", zorder=1)
             
    # L'état Lanczos est l'information principale. Alpha plus fort, zorder supérieur.
    plt.hist(el_valides, bins=bins, color="#457b9d", edgecolor="black", 
             linewidth=1.2, alpha=0.85, density=True, label=r"Dist. $E_L$ (Lanczos)", zorder=2)

    # 4. Tracé de la courbe KDE
    plt.plot(x_grid, kde_curve, color="#1d3557", linewidth=2.5, 
             label=r"KDE ($E_L$)", zorder=3)

    # 5. Lignes statistiques
    # Utilisation des f-strings avec le format LaTeX (r"...")
    plt.axvline(np.mean(results_e0), color="#e76f51", linestyle="--", linewidth=2, 
                label=fr"Mean $E_0$ ({np.mean(results_e0):.4f})", zorder=4)
                
    plt.axvline(np.mean(el_valides), color="#1d3557", linestyle="--", linewidth=2, 
                label=fr"Mean $E_L$ ({np.mean(el_valides):.4f})", zorder=4)
                
    plt.axvline(mode_EL, color="#8338ec", linestyle=":", linewidth=2.5, 
                label=fr"Mode $E_L$ ({mode_EL:.4f})", zorder=4)
                
    plt.axvline(exact_gs_energy, color="#2a9d8f", linestyle="-", linewidth=2.5, 
                label=r"Exact $E_0$", zorder=5)

    # 6. Mise en forme du graphique
    plt.xlabel(r"Energy per site ($E/L$)", fontsize=LABEL_FONTSIZE)
    plt.ylabel(r"Probability Density", fontsize=LABEL_FONTSIZE) 
    
    # ASTUCE PRO : ncol=2 permet de diviser la légende en 2 colonnes !
    # Avec 7 éléments, une seule colonne écraserait le graphique.
    plt.legend(loc="upper left", framealpha=0.9, edgecolor="black", 
               fontsize=LEGEND_FONTSIZE, ncol=2)
               
    plt.grid(True, linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()

    # 7. Sauvegarde
    if save_file:
        os.makedirs(data_file, exist_ok=True) 
        save_path = os.path.join(data_file, "distribution_energies_krylov.png")
        # N'oublie pas le bbox_inches='tight' pour éviter les labels coupés en PDF
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé sous : {save_path}")
    
    plt.show()
    
def plot_extrapolation_distribution(data_dict, mc_results, save_dir=save_default, name="extrapolation_distribution.png"):
    """
    Generates the histogram of the extrapolated energy distribution from the Bootstrap Monte Carlo.
    Optimized for LaTeX/Overleaf reports.
    """
    b_sim = mc_results["b_sim"]
    E_extrap_mean = mc_results["E_extrap_mean"]
    E_extrap_std = mc_results["E_extrap_std"]
    exact_gs_energy_per_site = data_dict["exact_gs_energy_per_site"]

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=fig_size, dpi=120)


    ax.hist(b_sim, bins=80, density=True, color='#fa5703', edgecolor='black', 
            linewidth=0.5, alpha=0.85, label=r"$E_0^{extrap}$ Dist.", zorder=2)

    ax.axvline(E_extrap_mean, color='#1d3557', linestyle='--', linewidth=2.5, 
               label=rf"$E_0^{{extrap}}$ mean", zorder=4)

    ax.axvspan(E_extrap_mean - E_extrap_std, E_extrap_mean + E_extrap_std, 
               color='#457b9d', alpha=0.25, label=r"$1\sigma$ Interval", zorder=1)
    
    ax.axvline(E_extrap_mean - E_extrap_std, color='#1d3557', linestyle=':', linewidth=1.5, alpha=0.6, zorder=3)
    ax.axvline(E_extrap_mean + E_extrap_std, color='#1d3557', linestyle=':', linewidth=1.5, alpha=0.6, zorder=3)

    ax.axvline(exact_gs_energy_per_site, color='green', linestyle='-', linewidth=2.5, 
               label=r"$E_0^{exact}$", zorder=5)

    ax.set_xlabel(r"Extrapolated Ground-State Energy per site ($E_0/L$)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"Probability Density", fontsize=LABEL_FONTSIZE)
    
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, framealpha=0.9, edgecolor="black")
    
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, name), dpi=300, bbox_inches='tight') 
    plt.show()
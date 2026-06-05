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
    Optimisé pour un rendu LaTeX/Overleaf : Zéro valeur numérique, code couleur strict.
    """
    # Calcul de la KDE pour lisser l'histogramme Lanczos
    kde = stats.gaussian_kde(el_valides)
    x_grid = np.linspace(np.min(el_valides), np.max(el_valides), 1000) 
    kde_curve = kde(x_grid)

    plt.figure(figsize=(8, 6), dpi=150) # Ajuste figsize selon tes constantes globales

    # 1. Tracé des histogrammes
    # VMC (Orange, transparent, en arrière-plan)
    plt.hist(results_e0, bins=bins, color="#f4a261", edgecolor="black", 
             alpha=0.4, density=True, label=r"VMC Distribution", zorder=1)
             
    # Lanczos (Bleu, bien visible, premier plan)
    plt.hist(el_valides, bins=bins, color="#457b9d", edgecolor="black", 
             linewidth=1.2, alpha=0.85, density=True, label=r"Lanczos Distribution", zorder=2)

    # Courbe KDE pour Lanczos (Tracée mais pas dans la légende pour alléger)
    plt.plot(x_grid, kde_curve, color="#1d3557", linewidth=2.5, zorder=3)

    # 2. Lignes statistiques (SANS valeurs numériques)
    plt.axvline(np.mean(results_e0), color="#e76f51", linestyle="--", linewidth=2.5, 
                label=r"Mean $E_{\mathrm{VMC}}$", zorder=4)
                
    plt.axvline(np.mean(el_valides), color="#1d3557", linestyle="--", linewidth=2.5, 
                label=r"Mean $E_L$", zorder=4)
                
    # Énergie exacte en NOIR comme demandé
    plt.axvline(exact_gs_energy, color="black", linestyle="-.", linewidth=2.5, 
                label=r"Exact $E_{\mathrm{ex}}$", zorder=5)

    # 3. Mise en forme du graphique
    plt.xlabel(r"Energy per site $E$", fontsize=14)
    plt.ylabel(r"Probability Density", fontsize=14) 
    
    # La légende est maintenant assez petite pour tenir sur une seule colonne
    plt.legend(loc="upper left", framealpha=0.9, edgecolor="black", fontsize=12)
               
    plt.grid(True, linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()

    # 4. Sauvegarde
    if save_file:
        os.makedirs(data_file, exist_ok=True) 
        save_path = os.path.join(data_file, "distribution_energies_krylov.png")
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
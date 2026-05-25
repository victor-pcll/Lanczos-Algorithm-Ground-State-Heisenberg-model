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
    Génère, affiche et sauvegarde la distribution des énergies pour les états initiaux 
    et après application de la méthode de Krylov/Lanczos.
    
    Arguments:
        el_valides (array-like): Énergies calculées après le pas de Lanczos.
        results_e0 (array-like): Énergies de l'état initial (VMC/RBM).
        exact_gs_energy (float): Valeur exacte de l'énergie de l'état fondamental.
        data_file (str): Chemin du dossier où sauvegarder l'image.
        bins (int, optionnel): Nombre de colonnes pour l'histogramme (défaut: 40).
    """
    
    # 1. Calcul du KDE (Kernel Density Estimation) pour EL
    kde = stats.gaussian_kde(el_valides)
    x_grid = np.linspace(np.min(el_valides), np.max(el_valides), 10000)
    kde_curve = kde(x_grid)
    mode_EL = x_grid[np.argmax(kde_curve)]

    # 2. Configuration de la figure
    plt.figure(figsize=(9, 5))

    # 3. Tracé des histogrammes (avec couleurs et labels distincts)
    plt.hist(el_valides, bins=bins, edgecolor="black", color="#457b9d", alpha=0.8, density=True, label=r"Distribution $E_L$ (Lanczos)")
    plt.hist(results_e0, bins=bins, edgecolor="black", color="#f4a261", alpha=0.6, density=True, label=r"Distribution $E_0$ (VMC/RBM)")

    # 4. Tracé des lignes statistiques
    plt.axvline(np.mean(el_valides), color="#e63946", linestyle="dashed", linewidth=2, label=fr"$E_L$ mean: {np.mean(el_valides):.5f}")
    plt.axvline(np.mean(results_e0), color="green", linestyle="dashed", linewidth=2, label=fr"$E_0$ mean: {np.mean(results_e0):.5f}")
    plt.axvline(exact_gs_energy, color="black", linestyle="dashed", linewidth=2, label=f"exact energy: {exact_gs_energy:.5f}")
    plt.axvline(mode_EL, color="#8338ec", linestyle="-", linewidth=2, label=f"Mode $E_L$: {mode_EL:.5f}")
    
    # 5. Tracé de la courbe KDE
    plt.plot(x_grid, kde_curve, color="black", linewidth=2.5, label="Density curve (KDE EL)")

    # 6. Mise en forme du graphique
    # plt.title("Distribution des Énergies (Méthode de Krylov)", fontsize=15, fontweight="bold")
    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Probability density", fontsize=12) # Corrigé car density=True
    plt.legend(loc="best", framealpha=0.9, edgecolor="black")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # 7. Sauvegarde et affichage
    if save_file:
        os.makedirs(data_file, exist_ok=True) 
        save_path = os.path.join(data_file, "distribution_energies_krylov.png")
        plt.savefig(save_path, dpi=300)
        print(f"Graphique sauvegardé sous : {save_path}")
    
    plt.show()

def plot_extrapolation_distribution(data_dict, mc_results, save_dir=save_default, name = "extrapolation_distribution.png"):
    """
    Génère l'histogramme de la distribution issue du Monte Carlo.
    """
    b_sim = mc_results["b_sim"]
    E_extrap_mean = mc_results["E_extrap_mean"]
    E_extrap_std = mc_results["E_extrap_std"]
    exact_gs_energy_per_site = data_dict["exact_gs_energy_per_site"]

    os.makedirs(save_dir, exist_ok=True)
    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=120)

    ax2.hist(b_sim, bins=80, density=True, color='red', edgecolor='black', linewidth=0.5, label="Expectation distribution")

    ax2.axvline(E_extrap_mean, color='blue', linestyle='--', linewidth=2, label="Mean")
    ax2.axvspan(E_extrap_mean - E_extrap_std, E_extrap_mean + E_extrap_std, color='blue', alpha=0.15, label=r"± $\sigma$")
    ax2.axvline(E_extrap_mean - E_extrap_std, color='blue', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axvline(E_extrap_mean + E_extrap_std, color='blue', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axvline(exact_gs_energy_per_site, color='green', linestyle='-', linewidth=2.5, label="Exact Energy")

    ax2.set_xlabel("Extrapolated Ground-State Energy per site", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
    # ax2.set_title("Extrapolation Uncertainty Profile", fontsize=14, fontweight="bold")
    ax2.legend(loc="best", fontsize=LEGEND_FONTSIZE, framealpha=0.9, edgecolor="black")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, name), dpi=300) 
    plt.show()
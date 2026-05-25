import matplotlib.pyplot as plt
import json
import numpy as np
import os
from scipy import stats
from src.MC_extrapol import run_monte_carlo_extrapolation

import json
import os
import numpy as np
import matplotlib.pyplot as plt

LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 14

def plot_VMC(data_file, exact_gs_energy, save_fig=False, fig_name="VMC_Optimization_Metrics.png"):
    
    plt.rcParams.update({
        "font.size": LABEL_FONTSIZE,                      # Taille de base plus lisible
        "axes.titlesize": 14,                 # Taille des titres
        "axes.labelsize": LABEL_FONTSIZE,                 # Taille des labels d'axes
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": LEGEND_FONTSIZE,
        "axes.linewidth": 1.2,                # Cadres légèrement plus épais
        "grid.alpha": 0.5                     # Transparence de la grille globale
    })

    with open(data_file) as f:
        data = json.load(f)
        
    iters_RBM = np.array(data["Energy"]["iters"])
    
    mean_data = data["Energy"]["Mean"]
    if isinstance(mean_data, dict) and "real" in mean_data:
        energy_RBM = np.array(mean_data["real"])
    else:
        energy_RBM = np.array(mean_data)
        
    variance_RBM = np.array(data["Energy"]["Variance"])
    acceptance_RBM = np.array(data["acceptance"]["value"])

    relative_error = np.abs(energy_RBM - exact_gs_energy) / np.abs(exact_gs_energy)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=150) 

    colors = {
        "energy": "#e63946",    # Red
        "exact": "#1d3557",     # Dark Blue
        "error": "#457b9d",     # Steel Blue
        "variance": "#2a9d8f",  # Teal
        "accept": "#f4a261"     # Orange
    }

    # --- Top Left: Energy Optimization ---
    axs[0, 0].plot(iters_RBM, energy_RBM, color=colors["energy"], linewidth=2, label="VMC Energy")
    axs[0, 0].axhline(y=exact_gs_energy, color=colors["exact"], linestyle="--", linewidth=2, label="Exact $E_0$")
    axs[0, 0].set_title("Ground State Energy Convergence")
    axs[0, 0].set_xlabel("VMC Iteration")
    axs[0, 0].set_ylabel(r"Energy $E$")
    axs[0, 0].legend(loc="upper right", framealpha=0.9, edgecolor="black")

    # --- Top Right: Relative Error (Log Scale) ---
    axs[0, 1].semilogy(iters_RBM, relative_error, color=colors["error"], linewidth=2, label="Relative Error")
    axs[0, 1].set_title("Relative Energy Error")
    axs[0, 1].set_xlabel("VMC Iteration")
    axs[0, 1].set_ylabel(r"$\Delta E / |E_{exact}|$")
    axs[0, 1].legend(loc="upper right", framealpha=0.9, edgecolor="black")

    # --- Bottom Left: Energy Variance ---
    axs[1, 0].plot(iters_RBM, variance_RBM, color=colors["variance"], linewidth=2, label=r"Variance $\sigma^2$")
    axs[1, 0].set_title("Energy Variance")
    axs[1, 0].set_xlabel("VMC Iteration")
    axs[1, 0].set_ylabel(r"$\langle H^2 \rangle - \langle H \rangle^2$")
    axs[1, 0].set_yscale("log") 
    axs[1, 0].legend(loc="upper right", framealpha=0.9, edgecolor="black")

    # --- Bottom Right: Acceptance Rate ---
    axs[1, 1].plot(iters_RBM, acceptance_RBM, color=colors["accept"], linewidth=2, label="Acceptance Rate")
    axs[1, 1].set_title("MCMC Acceptance Rate")
    axs[1, 1].set_xlabel("VMC Iteration")
    axs[1, 1].set_ylabel("Probability")
    # On force l'axe Y de l'acceptance entre 0 et 1 (ou légèrement au-dessus pour respirer)
    axs[1, 1].set_ylim([0, max(1.05, np.max(acceptance_RBM))]) 
    axs[1, 1].legend(loc="upper right", framealpha=0.9, edgecolor="black")

    for ax in axs.flat:
        ax.grid(True, which='major', linestyle="-", alpha=0.3)
        
        if ax.get_yscale() == 'log':
            ax.grid(True, which='minor', linestyle=":", alpha=0.2)
            
        ax.tick_params(axis='both', direction='in', top=True, right=True, which='both')

    plt.tight_layout() 
    
    if save_fig:
        os.makedirs(os.path.dirname(fig_name) or ".", exist_ok=True)
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved as {fig_name}")
        
    plt.show()
    plt.rcdefaults() 

    return True

def plot_energy_distribution(el_valides, results_e0, exact_gs_energy, data_file, bins=40):
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
    os.makedirs(data_file, exist_ok=True) # Crée le dossier s'il n'existe pas
    save_path = os.path.join(data_file, "distribution_energies_krylov.png")
    plt.savefig(save_path, dpi=300)
    print(f"Graphique sauvegardé sous : {save_path}")
    
    plt.show()

def plot_zero_variance_extrapolation(results_e0, results_sigma_e0, el_valides, results_sigma_eL, exact_gs_energy, L, data_file):
    """
    Calcule, affiche et sauvegarde le graphe d'extrapolation à variance nulle.
    
    Arguments:
        results_e0 (array-like): Échantillons d'énergie de l'état initial (VMC/RBM).
        results_sigma_e0 (array-like): Variances locales de l'état initial.
        el_valides (array-like): Échantillons d'énergie après l'étape de Lanczos.
        results_sigma_eL (array-like): Variances locales après l'étape de Lanczos.
        exact_gs_energy (float): Valeur exacte de l'énergie de l'état fondamental (système total).
        L (int): Taille du système (nombre de sites) pour normaliser par site.
        data_file (str): Chemin du répertoire où sauvegarder l'image.
    """
    
    # 1. Calculs statistiques par site
    x1 = np.mean(results_sigma_e0) / L
    x1_std = np.std(results_sigma_e0, ddof=1) / L
    y1 = np.mean(results_e0) / L
    y1_std = np.std(results_e0, ddof=1) / L

    x2 = np.mean(results_sigma_eL) / L
    x2_std = np.std(results_sigma_eL, ddof=1) / L
    y2 = np.mean(el_valides) / L
    y2_std = np.std(el_valides, ddof=1) / L

    # 2. Régression linéaire (fit) entre les moyennes
    a, b = np.polyfit([x1, x2], [y1, y2], 1)

    # 3. Préparation de la ligne d'extrapolation
    x_line = np.linspace(0, max(x1, x2) * 1.1, 100)
    y_line = a * x_line + b
    exact_energy_per_site = exact_gs_energy / L

    # 4. Configuration de la figure
    plt.figure(figsize=(9, 6), dpi=120)

    # Nuages de points (scatter) avec transparence
    plt.plot(results_sigma_e0 / L, results_e0 / L, "o", color="royalblue", markersize=3, alpha=0.2, label=r"$E_0$ Samples", zorder=1)
    plt.plot(results_sigma_eL / L, el_valides / L, "o", color="darkorange", markersize=3, alpha=0.2, label=r"$E_L$ Samples", zorder=1)
    
    # Barres d'erreur sur les moyennes
    plt.errorbar(x1, y1, xerr=x1_std, yerr=y1_std, fmt='o', color='darkblue', elinewidth=2, capsize=5, zorder=4)
    plt.errorbar(x2, y2, xerr=x2_std, yerr=y2_std, fmt='o', color='saddlebrown', elinewidth=2, capsize=5, zorder=4)
    
    # Lignes de fit et repères
    plt.plot(x_line, y_line, "k--", alpha=0.8, linewidth=1.5, label=f"Linear Fit (slope={a:.3f})", zorder=2)
    plt.plot(0, b, "r*", markersize=16, markeredgecolor='black', label=f"Extrapolated E = {b:.6f}", zorder=5)
    plt.axhline(y=exact_energy_per_site, color='forestgreen', linestyle='-', linewidth=2, alpha=0.8, label="Exact Energy", zorder=1)
    plt.axvline(x=0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, zorder=0)

    # 5. Mise en forme du graphique
    plt.xlabel("Variance $\sigma^2$ of energy per site", fontsize=13) 
    plt.ylabel("Energy $E$ per site", fontsize=13)
    # plt.title("Zero-Variance Extrapolation", fontsize=16, fontweight="bold")
    plt.legend(loc="best", fontsize=11, framealpha=0.9, edgecolor="black")
    plt.grid(True, linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()

    # 6. Sauvegarde et affichage
    os.makedirs(data_file, exist_ok=True)
    save_path = os.path.join(data_file, "extrapolation_plot.png")
    plt.savefig(save_path, dpi=150)
    print(f"Graphique d'extrapolation sauvegardé sous : {save_path}")
    plt.show()

    # 7. Sortie console
    print(f"\n--- Extrapolation Results (per site) ---")
    print(f"Extrapolated Energy (x=0) : {b:.8f}")
    print(f"Exact Energy              : {exact_energy_per_site:.8f}")
    print(f"Absolute Error            : {abs(b - exact_energy_per_site):.8e}")
    print(f"----------------------------------------\n")

def plot_master_extrapolation(data_dict, mc_results, L, save_dir="../../assets", name="master_extrapolation.png"):
    """
    Génère le graphique principal d'extrapolation (nuages de points et fit global).
    """
    toutes_les_simulations = data_dict["simulations"]
    exact_gs_energy_per_site = data_dict["exact_gs_energy_per_site"]
    E_extrap_mean = mc_results["E_extrap_mean"]
    E_extrap_std = mc_results["E_extrap_std"]
    
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(11, 7), dpi=120)
    cmap = plt.get_cmap('tab10')
    indices_sans_vert = [0, 1, 4, 5, 6, 7, 8, 9]
    colors = [cmap(indices_sans_vert[i % len(indices_sans_vert)]) for i in range(len(toutes_les_simulations))]

    all_x_means = []
    all_y_means = []

    for i, (nom_run, contenu) in enumerate(toutes_les_simulations.items()):
        data = contenu["data"]
        config = contenu["config"]
        
        results_e0 = np.array(data["Energy_E0"]["Mean"])
        results_sigma_e0 = np.array(data["Energy_E0"]["Variance"])
        el_valides = np.array(data["Energy_EL"]["Mean"])
        results_sigma_eL = np.array(data["Energy_EL"]["Variance"])

        x1 = np.mean(results_sigma_e0) / L
        y1 = np.mean(results_e0) / L
        x1_std = np.std(results_sigma_e0, ddof=1) / L
        y1_std = np.std(results_e0, ddof=1) / L

        x2 = np.mean(results_sigma_eL) / L
        y2 = np.mean(el_valides) / L
        x2_std = np.std(results_sigma_eL, ddof=1) / L
        y2_std = np.std(el_valides, ddof=1) / L

        all_x_means.extend([x1, x2])
        all_y_means.extend([y1, y2])

        lbl_e0_samples = r"$E_0$ Samples" if i == 0 else ""
        lbl_eL_samples = r"$E_L$ Samples" if i == 0 else ""
        plt.plot(results_sigma_e0 / L, results_e0 / L, "o", color=colors[i], markersize=2, alpha=0.05, label=lbl_e0_samples, zorder=1)
        plt.plot(results_sigma_eL / L, el_valides / L, "o", color=colors[i], markersize=2, alpha=0.05, label=lbl_eL_samples, zorder=1)

        run_label = f"Run {config['n_iter']} iters"
        plt.errorbar(x1, y1, xerr=x1_std, yerr=y1_std, fmt="o", color=colors[i], markeredgecolor="black", markersize=8, capsize=4, label=run_label + r" ($E_0$)", zorder=3)
        plt.errorbar(x2, y2, xerr=x2_std, yerr=y2_std, fmt="s", color=colors[i], markeredgecolor="black", markersize=8, capsize=4, label=run_label + r" ($E_L$)", zorder=3)

    # Global Fit
    a_global, b_global = np.polyfit(all_x_means, all_y_means, 1)
    x_line = np.linspace(0, max(all_x_means) * 1.1, 100)
    y_line = a_global * x_line + b_global

    plt.plot(x_line, y_line, "k--", alpha=0.9, linewidth=2, label=f"Global Fit", zorder=4)
    plt.errorbar(0, E_extrap_mean, yerr=E_extrap_std, fmt="o", color="red", markeredgecolor='black', markersize=6, capsize=3, capthick=2, elinewidth=2.5, label=f"Extrap. = {E_extrap_mean:.5f} ± {E_extrap_std:.5f}", zorder=5)
    plt.axhline(y=exact_gs_energy_per_site, color='green', linestyle='-', linewidth=2.5, alpha=0.8, label="Exact Energy", zorder=2)
    plt.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=0)
    
    plt.xlabel(r"Variance $\sigma^2$ of energy per site", fontsize=LABEL_FONTSIZE) 
    plt.ylabel(r"Energy $E$ per site", fontsize=LABEL_FONTSIZE)
    # plt.title("Master Zero-Variance Extrapolation (All Runs)", fontsize=16, fontweight="bold")
    plt.legend(loc='upper left', fontsize=LEGEND_FONTSIZE, framealpha=0.9, edgecolor="black")
    plt.grid(True, linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, name), dpi=300)
    plt.show()

def plot_extrapolation_distribution(data_dict, mc_results, save_dir="../../assets", name = "extrapolation_distribution.png"):
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
    ax2.legend(loc="upper left", fontsize=LEGEND_FONTSIZE, framealpha=0.9, edgecolor="black")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, name), dpi=300) 
    plt.show()

def plot_multiple_extrapolation_distributions(mc_results_dict, exact_gs_energy_per_site, save_dir="../../assets", name="extrapolation_distribution_comparison.png"):
    """
    Génère des histogrammes superposés pour comparer plusieurs distributions issues du Monte Carlo.
    
    Arguments:
        mc_results_dict (dict): Dictionnaire { "Nom du Run": mc_results }.
        exact_gs_energy_per_site (float): Énergie exacte de l'état fondamental par site.
        save_dir (str): Dossier où sauvegarder l'image.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=120)

    # Palette de couleurs personnalisée
    colors = ['#fa5703', '#457b9d', '#2a9d8f', '#9b5de5', '#e63946']

    # --- BOUCLE SUR CHAQUE RUN ---
    for i, (label, mc_results) in enumerate(mc_results_dict.items()):
        b_sim = mc_results["b_sim"]
        E_extrap_mean = mc_results["E_extrap_mean"]
        E_extrap_std = mc_results["E_extrap_std"]
        
        # Sélection cyclique de la couleur
        color = colors[i % len(colors)]

        # Histogramme (avec alpha=0.6 pour voir à travers)
        ax2.hist(b_sim, bins=80, density=True, color=color, alpha=0.4, label=f"{label} std: {E_extrap_std:.6f}")

        # Ligne de la moyenne
        ax2.axvline(E_extrap_mean, color=color, linestyle='--', linewidth=2, 
                    label=f"{label} Mean: {E_extrap_mean:.6f}")
        
        # Zone grisée pour l'écart-type (plus transparente pour ne pas surcharger)
        ax2.axvspan(E_extrap_mean - E_extrap_std, E_extrap_mean + E_extrap_std, 
                    color=color, alpha=0.1)

    # --- ÉNERGIE EXACTE (tracée une seule fois) ---
    ax2.axvline(exact_gs_energy_per_site, color='green', linestyle='-', 
                linewidth=2.5, label=f"Exact Energy: {exact_gs_energy_per_site:.6f}", zorder=10)

    # --- MISE EN FORME ---
    ax2.set_xlabel("Extrapolated Ground-State Energy per site", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
    ax2.legend(loc="upper left", fontsize=LEGEND_FONTSIZE, framealpha=0.9, edgecolor="black")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, name), dpi=300) 
    plt.show()

def plot_alpha_convergence(historique_alpha, alpha_min, alpha_err, 
                           data_file="../../assets/", tail_length=100):
    """
    """
    # ==========================================
    # 0. Configuration stricte pour LaTeX
    # ==========================================
    plt.rcParams.update({
        "axes.labelsize": 16,       # Grosse police pour les axes
        "xtick.labelsize": 16,      # Grosse police pour les graduations
        "ytick.labelsize": 16,
        "legend.fontsize": LEGEND_FONTSIZE,      # Grosse police pour la légende
        "axes.linewidth": 1.5,
        "grid.alpha": 0.5
    })

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
    plt.figure(figsize=(8, 6), dpi=150)
    
    plt.plot(historique_alpha, marker='o', markersize=4, color="#457b9d", linewidth=1.5, alpha=0.7, label=r"VMC $\alpha$")
    
    # Cible et Erreur Bootstrap
    plt.axhline(y=alpha_min, color="#2a9d8f", linestyle="--", linewidth=2.5, label=r"Target $\alpha_{opt}$")
    plt.axhspan(alpha_min - alpha_err, alpha_min + alpha_err, color="#2a9d8f", alpha=0.25)

    # Moyenne et Bruit VMC
    plt.axhline(y=alpha_mean, color="gray", linestyle=":", linewidth=2.5, label=r"VMC Mean")
    plt.axhspan(alpha_mean - alpha_std, alpha_mean + alpha_std, color='gray', alpha=0.15)

    plt.xlabel("VMC Iteration", fontsize=LABEL_FONTSIZE)
    plt.ylabel(r"Lanczos Parameter $\alpha$", fontsize=LABEL_FONTSIZE)
    plt.grid(True, linestyle="--")
    plt.legend(loc="best", framealpha=0.9, edgecolor="black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_file, "alpha_trajectory.pdf"), bbox_inches='tight')
    plt.show()

    # ==========================================
    # FIGURE 2 : Erreur Absolue (Échelle Log)
    # ==========================================
    plt.figure(figsize=(8, 6), dpi=150)
    
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
    plt.savefig(os.path.join(data_file, "alpha_error_log.pdf"), bbox_inches='tight')
    plt.show()

    # ==========================================
    # FIGURE 3 : Histogramme de la zone convergée
    # ==========================================
    plt.figure(figsize=(8, 6), dpi=150)
    
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
    plt.savefig(os.path.join(data_file, "alpha_histogram.pdf"), bbox_inches='tight')
    plt.show()

    # Nettoyage des paramètres pour ne pas affecter la suite de ton code
    plt.rcdefaults()
    
    return alpha_mean, alpha_std

def plot_energy_vs_frustration(data_dict, L, save_dir="../../assets", j2_key="J2/J1"):
    """
    Extrait les données de toutes les simulations et trace l'évolution 
    de l'énergie de l'état fondamental en fonction de la frustration (J2/J1).
    """
    j2_list, e0_list, el_list, exact_list, e0_err_list, el_err_list = [], [], [], [], [], []

    # 1. Extraction et calculs
    for nom_run, contenu in data_dict["simulations"].items():
        data = contenu["data"]
        config = contenu["config"]
        
        # Récupération sécurisée de J2 (par défaut 0.0 si non trouvé)
        j2 = config.get(j2_key, 0.0) 
        exact_gs = data["Exact_Energy"] / L
        
        # Moyennes des variances et énergies par site
        x1 = np.mean(data["Energy_E0"]["Variance"]) / L
        y1 = np.mean(data["Energy_E0"]["Mean"]) / L
        x2 = np.mean(data["Energy_EL"]["Variance"]) / L
        y2 = np.mean(data["Energy_EL"]["Mean"]) / L
        
        j2_list.append(j2)
        e0_list.append(y1)
        e0_err_list.append(x1)
        el_list.append(y2)
        el_err_list.append(x2)
        exact_list.append(exact_gs)

    # 2. Tri des données par valeur croissante de J2
    sorted_data = sorted(zip(j2_list, e0_list, el_list, exact_list))
    j2_list, e0_list, el_list, exact_list = map(np.array, zip(*sorted_data))

    # 3. Tracé du graphique principal (Énergie)
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6), dpi=120)

    plt.errorbar(j2_list, e0_list, yerr=e0_err_list, fmt='o', ecolor='#457b9d', capsize=2, label='VMC pur ($E_0$)')
    plt.errorbar(j2_list, el_list, yerr=el_err_list, fmt='s', ecolor='#f4a261', capsize=2, label='1 pas de Lanczos ($E_L$)')
    # plt.plot(j2_list, exact_list, 'k--', linewidth=2.5, alpha=0.8, label='Énergie Exacte')
    plt.scatter(j2_list, exact_list, color='black', s=25, marker='+', label='Exact (points)', zorder=5)

    plt.xlabel(r'Ratio de frustration $J_2 / J_1$', fontsize=14)
    plt.ylabel(r'Énergie fondamentale par site $E/L$', fontsize=14)
    #  plt.title('Impact de la frustration quantique sur l\'énergie', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12, framealpha=0.9, edgecolor="black")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Point de Majumdar-Ghosh
    # plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    # plt.text(0.51, min(exact_list), 'Frustration\nMaximale (0.5)', color='gray', fontsize=10, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "energy_vs_frustration.png"), dpi=300)
    plt.show()

def analyze_and_plot_all_j2(data_dict, L, N_sim=5000, save_dir="./assets", j2_key="J2/J1", title="frustrated"):
    """
    1. Groups data by J2/J1 ratio.
    2. Runs Monte Carlo extrapolation and individual plots for each J2.
    3. Plots the final global graphs (Energy vs. J2/J1 and Energy Error vs. J2/J1).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # =========================================================
    # STEP 1: Group data by J2 value
    # =========================================================
    j2_groups = {}
    
    for run_name, content in data_dict["simulations"].items():
        data = content["data"]
        config = content["config"]
        j2 = config.get(j2_key, 0.0)
        
        if j2 not in j2_groups:
            j2_groups[j2] = {
                "X_means": [], "Y_means": [],
                "X_stds": [], "Y_stds": [],
                "exact_gs_energy_per_site": data["Exact_Energy"] / L
            }
            
        # Means and standard deviations per site for this run
        x1_m = np.mean(data["Energy_E0"]["Variance"]) / L
        y1_m = np.mean(data["Energy_E0"]["Mean"]) / L
        x1_s = np.std(data["Energy_E0"]["Variance"], ddof=1) / L
        y1_s = np.std(data["Energy_E0"]["Mean"], ddof=1) / L
        
        x2_m = np.mean(data["Energy_EL"]["Variance"]) / L
        y2_m = np.mean(data["Energy_EL"]["Mean"]) / L
        x2_s = np.std(data["Energy_EL"]["Variance"], ddof=1) / L
        y2_s = np.std(data["Energy_EL"]["Mean"], ddof=1) / L

        j2_groups[j2]["X_means"].extend([x1_m, x2_m])
        j2_groups[j2]["Y_means"].extend([y1_m, y2_m])
        j2_groups[j2]["X_stds"].extend([x1_s, x2_s])
        j2_groups[j2]["Y_stds"].extend([y1_s, y2_s])


    # =========================================================
    # STEP 2: Run VMC and individual plots per J2
    # =========================================================
    j2_list, e0_list, e0_err_list = [], [], []
    el_list, el_err_list = [], []
    extrap_list, extrap_err_list = [], []
    exact_list = []
    
    for j2, group in j2_groups.items():
        print(f"\n➤ Processing for J2/J1 = {j2} ...")
        
        # Prepare specific dictionary for this J2
        mini_data_dict = {
            "X_means": np.array(group["X_means"]),
            "Y_means": np.array(group["Y_means"]),
            "X_stds": np.array(group["X_stds"]),
            "Y_stds": np.array(group["Y_stds"]),
            "exact_gs_energy_per_site": group["exact_gs_energy_per_site"]
        }
        
        # Run extrapolation
        mc_results = run_monte_carlo_extrapolation(mini_data_dict, N_sim, verbose=False)
        
        # Generate specific plots (replace '.' with 'p' in filenames to avoid path issues)
        j2_str = f"{j2:.2f}".replace('.', 'p')
        # plot_master_extrapolation(mini_data_dict, mc_results, L, save_dir=save_dir, name=f"AL_extrap_J2_{j2_str}")
        # plot_extrapolation_distribution(mini_data_dict, mc_results, save_dir=save_dir, name=f"AL_distrib_J2_{j2_str}")
        
        # Store for the final plot
        j2_list.append(j2)
        e0_list.append(np.mean(group["Y_means"][0::2]))  # Mean of E0 for this group
        e0_err_list.append(np.mean(group["Y_stds"][0::2]))
        el_list.append(np.mean(group["Y_means"][1::2]))  # Mean of EL for this group
        el_err_list.append(np.mean(group["Y_stds"][1::2]))
        extrap_list.append(mc_results["E_extrap_mean"])
        extrap_err_list.append(mc_results["E_extrap_std"])
        exact_list.append(group["exact_gs_energy_per_site"])


    # =========================================================
    # STEP 3: Sort and create the final global plots
    # =========================================================
    sorted_data = sorted(zip(j2_list, e0_list, e0_err_list, el_list, el_err_list, extrap_list, extrap_err_list, exact_list))
    j2_list, e0_list, e0_err_list, el_list, el_err_list, extrap_list, extrap_err_list, exact_list = map(np.array, zip(*sorted_data))

    # --- Plot 1: Absolute Energies ---
    plt.figure(figsize=(10, 6), dpi=120)

    # Plot with error bars (capsize adds small caps at the end of the bars)
    plt.errorbar(j2_list, e0_list, yerr=e0_err_list, fmt='o-', color='#457b9d', capsize=3, linewidth=1.5, label='Pure VMC ($E_0$)')
    plt.errorbar(j2_list, el_list, yerr=el_err_list, fmt='s-', color='#f4a261', capsize=3, linewidth=1.5, label='1 Lanczos step ($E_L$)')
    
    # MC Extrapolation
    plt.errorbar(j2_list, extrap_list, yerr=extrap_err_list, fmt='*-', color='#e63946', markersize=12, capsize=4, linewidth=2, label=r'MC Extrapolation $\sigma^2 \to 0$')
    
    # Exact energy
    plt.plot(j2_list, exact_list, 'k--', linewidth=2.5, alpha=0.8, label='Exact Energy')

    # Formatting
    plt.xlabel(r'Frustration Ratio $J_2/J_1$', fontsize=14)
    plt.ylabel(r'Ground-state energy per site $E/L$', fontsize=14)
    # plt.title('Impact of Frustration and Monte Carlo Extrapolation', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor="black")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, title), dpi=300)
    plt.show()

    # --- Plot 2: Energy Difference ---
    plt.figure(figsize=(10, 6), dpi=120)
    
    plt.errorbar(j2_list, extrap_list - exact_list, yerr=extrap_err_list, fmt='*-', color='#e63946', markersize=12, capsize=4, linewidth=2, label=r'MC Extrapolation $\sigma^2 \to 0$')

    # Formatting
    plt.xlabel(r'Frustration Ratio $J_2/J_1$', fontsize=14)
    plt.ylabel(r'Energy Difference per site $(E - E_{exact})/L$', fontsize=14)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.title('Extrapolation Error vs. Frustration', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor="black")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "global_energy_diff_vs_frustration.png"), dpi=300) # Fixed filename
    plt.show()

    print(f"\n✅ Analysis complete. All plots have been saved to: {save_dir}")

def plot_VMC_sequential(data_file_init, data_file_alpha, exact_gs_energy, save_fig=False, fig_name="VMC_Sequential_Metrics.png"):
    """
    Trace l'évolution continue de l'optimisation VMC en deux phases :
    1. VMC pur (Trait plein)
    2. Optimisation avec Lanczos alpha (Traits pointillés)
    """
    
    plt.rcParams.update({
        "font.size": LABEL_FONTSIZE,
        "axes.titlesize": 14,
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": LEGEND_FONTSIZE,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.5
    })

    # --- Fonction interne pour extraire proprement les logs ---
    def load_and_extract(file_path):
        with open(file_path) as f:
            data = json.load(f)
            
        iters = np.array(data["Energy"]["iters"])
        mean_data = data["Energy"]["Mean"]
        
        if isinstance(mean_data, dict) and "real" in mean_data:
            energy = np.array(mean_data["real"])
        else:
            energy = np.array(mean_data)
            
        variance = np.array(data["Energy"]["Variance"])
        acceptance = np.array(data["acceptance"]["value"])
        
        error = np.abs(energy - exact_gs_energy) / np.abs(exact_gs_energy)
        return iters, energy, variance, acceptance, error

    # 1. Extraction des deux phases
    iters1, eng1, var1, acc1, err1 = load_and_extract(data_file_init)
    iters2, eng2, var2, acc2, err2 = load_and_extract(data_file_alpha)

    # 2. Décalage de l'axe X pour que la phase 2 suive directement la phase 1
    # On soustrait le premier iter de la phase 2 et on ajoute le dernier de la phase 1
    iters2_shifted = iters2 - iters2[0] + iters1[-1]

    # 3. Préparation du Graphique
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=150) 

    colors = {
        "energy": "#e63946",    
        "exact": "#1d3557",     
        "error": "#457b9d",     
        "variance": "#2a9d8f",  
        "accept": "#fa5703"     # Orange foncé (Ton code couleur)
    }

    # --- Fonction utilitaire pour tracer les deux phases ---
    def plot_phases(ax, y1, y2, color, ylabel, label_base, is_log=False):
        # Phase 1 : VMC Init (Plein)
        ax.plot(iters1, y1, color=color, linewidth=2, linestyle='-', label=f"{label_base} (Init)")
        # Phase 2 : VMC Alpha (Pointillés)
        ax.plot(iters2_shifted, y2, color=color, linewidth=2.5, linestyle='--', alpha=0.9, label=f"{label_base} ($\\alpha$ opt)")
        
        # Ligne de démarcation
        ax.axvline(x=iters1[-1], color='gray', linestyle=':', linewidth=1.5, alpha=0.8, zorder=0)
        
        ax.set_xlabel("VMC Iteration")
        ax.set_ylabel(ylabel)
        if is_log:
            ax.set_yscale("log")

    # --- Top Left: Energy ---
    plot_phases(axs[0, 0], eng1, eng2, colors["energy"], r"Energy $E$", "VMC Energy")
    axs[0, 0].axhline(y=exact_gs_energy, color=colors["exact"], linestyle="-.", linewidth=2, label="Exact $E_0$", zorder=0)
    axs[0, 0].set_title("Ground State Energy Convergence")
    axs[0, 0].legend(loc="upper right", framealpha=0.9, edgecolor="black")

    # --- Top Right: Relative Error ---
    plot_phases(axs[0, 1], err1, err2, colors["error"], r"$\Delta E / |E_{exact}|$", "Rel. Error", is_log=True)
    axs[0, 1].set_title("Relative Energy Error")
    axs[0, 1].legend(loc="upper right", framealpha=0.9, edgecolor="black")

    # --- Bottom Left: Variance ---
    plot_phases(axs[1, 0], var1, var2, colors["variance"], r"$\langle H^2 \rangle - \langle H \rangle^2$", "Variance", is_log=True)
    axs[1, 0].set_title("Energy Variance")
    axs[1, 0].legend(loc="upper right", framealpha=0.9, edgecolor="black")

    # --- Bottom Right: Acceptance ---
    plot_phases(axs[1, 1], acc1, acc2, colors["accept"], "Probability", "Accept. Rate")
    axs[1, 1].set_title("MCMC Acceptance Rate")
    max_acc = max(np.max(acc1), np.max(acc2))
    axs[1, 1].set_ylim([0, max(1.05, max_acc)]) 
    axs[1, 1].legend(loc="upper right", framealpha=0.9, edgecolor="black")

    # --- Formatage Global ---
    for ax in axs.flat:
        ax.grid(True, which='major', linestyle="-", alpha=0.3)
        if ax.get_yscale() == 'log':
            ax.grid(True, which='minor', linestyle=":", alpha=0.2)
        ax.tick_params(axis='both', direction='in', top=True, right=True, which='both')

    plt.tight_layout() 
    
    if save_fig:
        os.makedirs(os.path.dirname(fig_name) or ".", exist_ok=True)
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved as {fig_name}")
        
    plt.show()
    plt.rcdefaults() 

    return True
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from .plot_params import *
from ..MC_extrapol import run_monte_carlo_extrapolation

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
    plt.figure(figsize=fig_size, dpi=120)

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

def plot_multiple_extrapolation_distributions(mc_results_dict, exact_gs_energy_per_site, save_dir=save_default, name="extrapolation_distribution_comparison.png"):
    """
    Génère des histogrammes superposés pour comparer plusieurs distributions issues du Monte Carlo.
    
    Arguments:
        mc_results_dict (dict): Dictionnaire { "Nom du Run": mc_results }.
        exact_gs_energy_per_site (float): Énergie exacte de l'état fondamental par site.
        save_dir (str): Dossier où sauvegarder l'image.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig2, ax2 = plt.subplots(figsize=fig_size, dpi=120)

    colors = ['#fa5703', '#457b9d', '#2a9d8f', '#9b5de5', '#e63946']

    for i, (label, mc_results) in enumerate(mc_results_dict.items()):
        b_sim = mc_results["b_sim"]
        E_extrap_mean = mc_results["E_extrap_mean"]
        E_extrap_std = mc_results["E_extrap_std"]
        
        color = colors[i % len(colors)]

        ax2.hist(b_sim, bins=80, density=True, color=color, alpha=0.4, 
                label=r"$E_{0, \mathrm{" + label + r"}}^{\mathrm{extrap}}$ Distrib.")

        ax2.axvline(E_extrap_mean, color=color, linestyle='--', linewidth=2, 
                    label=f"{label} Mean: {E_extrap_mean:.6f}")
        
        ax2.axvspan(E_extrap_mean - E_extrap_std, E_extrap_mean + E_extrap_std, 
                    color=color, alpha=0.1)

    ax2.axvline(exact_gs_energy_per_site, color='green', linestyle='-', 
                linewidth=2.5, label=r"$E_0^{exact}$")

    ax2.set_xlabel("Extrapolated Ground-State Energy per site", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
    ax2.legend(loc="best", fontsize=LEGEND_FONTSIZE, framealpha=0.9, edgecolor="black")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, name), dpi=300) 
    plt.show()

def plot_master_extrapolation(data_dict, mc_results, L, save_dir=save_default, name="master_extrapolation.png"):
    """
    Génère le graphique principal d'extrapolation (nuages de points et fit global).
    """
    toutes_les_simulations = data_dict["simulations"]
    exact_gs_energy_per_site = data_dict["exact_gs_energy_per_site"]
    E_extrap_mean = mc_results["E_extrap_mean"]
    E_extrap_std = mc_results["E_extrap_std"]
    
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=fig_size, dpi=120)
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

        lbl_e0_samples = r"$E_{VMC}$" if i == 0 else ""
        lbl_eL_samples = r"$E_L$" if i == 0 else ""
        
        plt.plot(results_sigma_e0 / L, results_e0 / L, "o", color=colors[i], markersize=2, alpha=0.05, label=lbl_e0_samples, zorder=1)
        plt.plot(results_sigma_eL / L, el_valides / L, "s", color=colors[i], markersize=2, alpha=0.05, label=lbl_eL_samples, zorder=1)

        run_id = f"R{config['n_iter']}"
        plt.errorbar(x1, y1, xerr=x1_std, yerr=y1_std, fmt="o", color=colors[i], markeredgecolor="black", markersize=8, capsize=4, label=fr"$E_{{VMC}}$ ({run_id})", zorder=3)
        plt.errorbar(x2, y2, xerr=x2_std, yerr=y2_std, fmt="s", color=colors[i], markeredgecolor="black", markersize=8, capsize=4, label=fr"$E_L$ ({run_id})", zorder=3)

    a_global, b_global = np.polyfit(all_x_means, all_y_means, 1)
    x_line = np.linspace(0, max(all_x_means) * 1.1, 100)
    y_line = a_global * x_line + b_global

    plt.plot(x_line, y_line, "k--", alpha=0.9, linewidth=2, label="Global Fit", zorder=4)
    
    extrap_label = r"$E_0^{\mathrm{extrap}}$"
    plt.errorbar(0, E_extrap_mean, yerr=E_extrap_std, fmt="o", color="red", markeredgecolor='black', markersize=6, capsize=3, capthick=2, elinewidth=2.5, label=extrap_label, zorder=5)
    
    plt.axhline(y=exact_gs_energy_per_site, color='green', linestyle='-', linewidth=2.5, alpha=0.8, label=r"$E_0^{exact}$", zorder=2)
    plt.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=0)
    
    plt.xlabel(r"Variance $\sigma^2$ of energy per site", fontsize=LABEL_FONTSIZE) 
    plt.ylabel(r"Energy $E$ per site", fontsize=LABEL_FONTSIZE)
    
    plt.grid(True, linestyle="--", alpha=0.5, zorder=0)

    plt.legend(
        loc='center left',            # Point d'ancrage de la légende
        bbox_to_anchor=(1.02, 0.5),   # (x, y) : x=1.02 la pousse juste à droite des axes, y=0.5 la centre verticalement
        fontsize=11,         
        ncol=1,                       # On repasse à 1 colonne puisque l'espace horizontal n'est plus limité par le cadre
        labelspacing=0.6,     
        handletextpad=0.5,    
        framealpha=1.0,               # Pas besoin de transparence si elle est dehors
        edgecolor="black"
    )

    plt.savefig(os.path.join(save_dir, name), dpi=300, bbox_inches='tight')
    plt.show()

def plot_individual_zero_variance_j2(data_dict, L, N_sim=10000, save_dir="../../assets/extrapolations_j2", show=False):
    """
    Génère un graphique d'extrapolation séparé pour chaque ratio J2/J1.
    Inclut les barres d'erreurs sur l'énergie (Y) ET sur la variance (X).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # =========================================================
    # 1. Groupement des données par J2
    # =========================================================
    j2_groups = {}
    for run_name, content in data_dict["simulations"].items():
        data = content["data"]
        config = content["config"]
        j2 = config.get("J2/J1", 0.0)
        
        if j2 not in j2_groups:
            j2_groups[j2] = {
                "X_means": [], "Y_means": [],
                "X_stds": [], "Y_stds": [],
                "exact_gs_energy_per_site": data["Exact_Energy"] / L
            }
            
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
    # 2. Création d'un graphique par J2
    # =========================================================
    for j2, group in j2_groups.items():
        print(f"➤ Génération du graphique pour J2/J1 = {j2} ...")
        
        # Extraction des tableaux pour ce J2
        x0_means = np.array(group["X_means"][0::2])
        y0_means = np.array(group["Y_means"][0::2])
        x0_stds  = np.array(group["X_stds"][0::2])
        y0_stds  = np.array(group["Y_stds"][0::2])
        
        xl_means = np.array(group["X_means"][1::2])
        yl_means = np.array(group["Y_means"][1::2])
        xl_stds  = np.array(group["X_stds"][1::2])
        yl_stds  = np.array(group["Y_stds"][1::2])
        
        exact_gs = group["exact_gs_energy_per_site"]

        # Extrapolation globale Monte Carlo pour ce J2
        mini_data_dict = {
            "X_means": np.array(group["X_means"]), "Y_means": np.array(group["Y_means"]),
            "X_stds": np.array(group["X_stds"]), "Y_stds": np.array(group["Y_stds"]),
            "exact_gs_energy_per_site": exact_gs
        }
        
        # NOTE: Assure-toi que run_monte_carlo_extrapolation est bien importée/définie
        mc_results = run_monte_carlo_extrapolation(mini_data_dict, N_sim, verbose=False)
        ex_mean = mc_results["E_extrap_mean"]
        ex_std = mc_results["E_extrap_std"]

        # --- Début du tracé ---
        plt.figure(figsize=fig_size, dpi=120)

        # 1. Tracé des points VMC avec barres d'erreurs (X et Y)
        plt.errorbar(x0_means, y0_means, xerr=x0_stds, yerr=y0_stds, 
                     fmt='o', color='#457b9d', alpha=0.8, markersize=6, capsize=3, 
                     linewidth=1.5, label='Pure VMC ($E_0$)')
        
        # 2. Tracé des points Lanczos avec barres d'erreurs (X et Y)
        plt.errorbar(xl_means, yl_means, xerr=xl_stds, yerr=yl_stds, 
                     fmt='s', color='#f4a261', alpha=0.8, markersize=6, capsize=3, 
                     linewidth=1.5, label='1 Lanczos step ($E_L$)')

        # 3. Ligne de tendance globale (relie le centre de masse des nuages au point V=0)
        max_x0, max_y0 = np.max(x0_means), np.max(y0_means)
        plt.plot([0, max_x0], [ex_mean, max_y0], color='#e63946', linestyle='--', linewidth=2, alpha=0.7)

        # 4. Point extrapolé à V=0 (avec barre d'erreur Y)
        plt.errorbar([0], [ex_mean], yerr=[ex_std], 
                     fmt='*', color='#e63946', markersize=16, capsize=4, 
                     linewidth=2.5, zorder=5, label='MC Extrapolation ($V=0$)')

        # 5. Énergie Exacte
        plt.scatter([0], [exact_gs], color='black', marker='+', s=150, linewidths=2.5, 
                    zorder=6, label='Exact Energy')

        # --- Formatage et Esthétique ---
        plt.axvline(x=0, color='black', linewidth=1.5, alpha=0.8, zorder=1)
        
        # Ajustement des limites de l'axe X pour bien voir le point V=0 et respirer à droite
        max_var = np.max(x0_means + x0_stds)
        plt.xlim(-0.05 * max_var, max_var * 1.1)
        
        plt.xlabel(r'Energy Variance per site $\sigma^2/L$', fontsize=14)
        plt.ylabel(r'Ground-state Energy per site $E/L$', fontsize=14)
        plt.title(f'Zero-Variance Extrapolation (Frustration Ratio $J_2/J_1 = {j2}$)', fontsize=16, fontweight='bold')
        
        plt.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor="black")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Sauvegarde propre
        j2_str = f"{j2:.2f}".replace('.', 'p')
        filename = os.path.join(save_dir, f"extrap_J2_{j2_str}.png")
        plt.savefig(filename, dpi=300)
        
        if show :
            plt.show()
        else:
            plt.close()

    print(f"✅ Tous les graphiques individuels ont été sauvegardés dans : {save_dir}")
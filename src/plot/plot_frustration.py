import matplotlib.pyplot as plt
from .plot_params import *
import numpy as np
import os

def plot_energy_vs_frustration(data_dict, L, save_dir=save_default, j2_key="J2/J1"):
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
    plt.figure(figsize=fig_size, dpi=120)

    plt.errorbar(j2_list, e0_list, yerr=e0_err_list, fmt='o', ecolor='#457b9d', capsize=2, label='VMC pur ($E_0$)')
    plt.errorbar(j2_list, el_list, yerr=el_err_list, fmt='s', ecolor='#f4a261', capsize=2, label='1 pas de Lanczos ($E_L$)')
    # plt.plot(j2_list, exact_list, 'k--', linewidth=2.5, alpha=0.8, label='Énergie Exacte')
    plt.scatter(j2_list, exact_list, color='black', s=25, marker='+', label='Exact (points)', zorder=5)

    plt.xlabel(r'Ratio de frustration $J_2 / J_1$', fontsize=LABEL_FONTSIZE)
    plt.ylabel(r'Énergie fondamentale par site $E/L$', fontsize=LABEL_FONTSIZE)
    #  plt.title('Impact de la frustration quantique sur l\'énergie', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=LEGEND_FONTSIZE, framealpha=0.9, edgecolor="black")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Point de Majumdar-Ghosh
    # plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    # plt.text(0.51, min(exact_list), 'Frustration\nMaximale (0.5)', color='gray', fontsize=10, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "energy_vs_frustration.png"), dpi=300)
    plt.show()
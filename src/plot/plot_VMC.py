import matplotlib.pyplot as plt
import json
import numpy as np
import os
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from .plot_params import *

def plot_VMC(data_file, exact_gs_energy, save_fig=False, fig_name="VMC_Optimization_Metrics.png"):

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

    fig, axs = plt.subplots(2, 2, figsize=fig_size, dpi=150) 

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

def plot_VMC_sequential(data_file_init, data_file_alpha, exact_gs_energy, save_fig=False, fig_name="VMC_Sequential_Metrics.png"):
    """
    Trace l'évolution continue de l'optimisation VMC en deux phases :
    1. VMC pur (Trait plein)
    2. Optimisation avec Lanczos alpha (Traits pointillés)
    """

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
    fig, axs = plt.subplots(2, 2, figsize=fig_size, dpi=150) 

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
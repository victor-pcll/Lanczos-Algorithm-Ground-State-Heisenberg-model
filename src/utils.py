import os
import json
import numpy as np
import pandas as pd

def save_simulation_logs(target_dir, n_iterations, e0_array, sigma_e0_array, el_array, sigma_el_array, time_array, exact_energy):
    """
    Formate et sauvegarde les résultats de la simulation dans un fichier JSON.
    Convertit automatiquement les arrays (Numpy/JAX) en listes Python pour la compatibilité JSON.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    iters_list = list(range(n_iterations))
    
    def to_list(arr):
        return np.array(arr).tolist()

    log_data = {
        "Energy_E0": {
            "iters": iters_list,
            "Mean": to_list(e0_array),
            "Variance": to_list(sigma_e0_array)
        },
        "Energy_EL": {
            "iters": iters_list,
            "Mean": to_list(el_array),
            "Variance": to_list(sigma_el_array)
        },
        "Time": to_list(time_array),
        "Exact_Energy": float(exact_energy) 
    }
    
    file_path = os.path.join(target_dir, "simulation_results.json")
    
    with open(file_path, "w") as f:
        json.dump(log_data, f, indent=4) # indent=4 rend le fichier lisible par un humain
        
    print(f"Simulation logs successfully saved to: {file_path}")

def load_simulation_data(dossiers, L, verbose=True):
    """
    Parcourt les dossiers, lit les fichiers JSON et extrait les statistiques.
    """
    toutes_les_simulations = {}
    X_means_list, X_stds_list = [], []
    Y_means_list, Y_stds_list = [], []
    exact_gs_energy_per_site = None

    if verbose:
        print(f"Loading and processing {len(dossiers)} folders...")

    for dossier in dossiers:
        fichier_config = os.path.join(dossier, "hyperparameters.json")
        fichier_resultats = os.path.join(dossier, "simulation_results.json")
        
        if os.path.exists(fichier_config) and os.path.exists(fichier_resultats):
            with open(fichier_config, "r") as f:
                config = json.load(f)
            with open(fichier_resultats, "r") as f:
                data = json.load(f)
                
            n_iter = config.get("n_iter", "inconnu")
            nom_run = f"Run_{n_iter}"
            toutes_les_simulations[nom_run] = {
                "config": config,
                "data": data
            }
            
            exact_gs_energy_per_site = data["Exact_Energy"] / L
            
            # Extraction E0
            results_e0 = np.array(data["Energy_E0"]["Mean"])
            results_sigma_e0 = np.array(data["Energy_E0"]["Variance"])
            x1_m = np.mean(results_sigma_e0) / L
            y1_m = np.mean(results_e0) / L
            x1_s = np.std(results_sigma_e0, ddof=1) / L
            y1_s = np.std(results_e0, ddof=1) / L
            
            # Extraction EL
            el_valides = np.array(data["Energy_EL"]["Mean"])
            results_sigma_eL = np.array(data["Energy_EL"]["Variance"])
            x2_m = np.mean(results_sigma_eL) / L
            y2_m = np.mean(el_valides) / L
            x2_s = np.std(results_sigma_eL, ddof=1) / L
            y2_s = np.std(el_valides, ddof=1) / L

            X_means_list.extend([x1_m, x2_m])
            Y_means_list.extend([y1_m, y2_m])
            X_stds_list.extend([x1_s, x2_s])
            Y_stds_list.extend([y1_s, y2_s])

    if verbose:
        print(f"✅ Loading complete: {len(toutes_les_simulations)} simulations processed!")

    # --- NOUVELLE SÉCURITÉ ---
    if len(toutes_les_simulations) == 0:
        raise ValueError("ERREUR : Aucune simulation n'a été chargée ! "
                         "Vérifie que les dossiers existent et contiennent bien les fichiers JSON.")
    # -------------------------

    return {
        "simulations": toutes_les_simulations,
        "X_means": np.array(X_means_list),
        "Y_means": np.array(Y_means_list),
        "X_stds": np.array(X_stds_list),
        "Y_stds": np.array(Y_stds_list),
        "exact_gs_energy_per_site": exact_gs_energy_per_site
    }

def load_simulation_data_J1J2(dossiers, L, verbose=True):
    """
    Parcourt les dossiers, lit les fichiers JSON et extrait les statistiques.
    """
    toutes_les_simulations = {}
    X_means_list, X_stds_list = [], []
    Y_means_list, Y_stds_list = [], []
    exact_gs_energy_per_site = None

    if verbose:
        print(f"Loading and processing {len(dossiers)} folders...")

    for dossier in dossiers:
        fichier_config = os.path.join(dossier, "hyperparameters.json")
        fichier_resultats = os.path.join(dossier, "simulation_results.json")
        
        if os.path.exists(fichier_config) and os.path.exists(fichier_resultats):
            with open(fichier_config, "r") as f:
                config = json.load(f)
            with open(fichier_resultats, "r") as f:
                data = json.load(f)
                
            # CORRECTION 1 : Utiliser le nom du dossier comme clé pour éviter les écrasements
            nom_run = os.path.basename(os.path.normpath(dossier))
            
            toutes_les_simulations[nom_run] = {
                "config": config,
                "data": data
            }
            
            exact_gs_energy_per_site = data["Exact_Energy"] / L
            
            # Extraction E0
            results_e0 = np.array(data["Energy_E0"]["Mean"])
            results_sigma_e0 = np.array(data["Energy_E0"]["Variance"])
            x1_m = np.mean(results_sigma_e0) / L
            y1_m = np.mean(results_e0) / L
            x1_s = np.std(results_sigma_e0, ddof=1) / L
            y1_s = np.std(results_e0, ddof=1) / L
            
            # Extraction EL
            el_valides = np.array(data["Energy_EL"]["Mean"])
            results_sigma_eL = np.array(data["Energy_EL"]["Variance"])
            x2_m = np.mean(results_sigma_eL) / L
            y2_m = np.mean(el_valides) / L
            x2_s = np.std(results_sigma_eL, ddof=1) / L
            y2_s = np.std(el_valides, ddof=1) / L

            X_means_list.extend([x1_m, x2_m])
            Y_means_list.extend([y1_m, y2_m])
            X_stds_list.extend([x1_s, x2_s])
            Y_stds_list.extend([y1_s, y2_s])
            
        # CORRECTION 2 : Avertir si des fichiers manquent
        else:
            if verbose:
                print(f"⚠️  Dossier ignoré (fichiers JSON manquants) : {dossier}")

    if verbose:
        print(f"✅ Loading complete: {len(toutes_les_simulations)} simulations processed!")

    if len(toutes_les_simulations) == 0:
        raise ValueError("ERREUR : Aucune simulation n'a été chargée ! "
                         "Vérifie que les dossiers existent et contiennent bien les fichiers JSON.")

    return {
        "simulations": toutes_les_simulations,
        "X_means": np.array(X_means_list),
        "Y_means": np.array(Y_means_list),
        "X_stds": np.array(X_stds_list),
        "Y_stds": np.array(Y_stds_list),
        "exact_gs_energy_per_site": exact_gs_energy_per_site
    }

def create_results_table(data_dict, L):
    table_data = []
    
    # Parcourir chaque simulation chargée dans le dictionnaire
    for run_name, sim_info in data_dict["simulations"].items():
        config = sim_info["config"]
        data = sim_info["data"]
        
        # 1. vmc_it
        vmc_it = config.get("n_iter", run_name)
        
        # Extraction des arrays
        e0_array = np.array(data["Energy_E0"]["Mean"])
        var0_array = np.array(data["Energy_E0"]["Variance"])
        el_array = np.array(data["Energy_EL"]["Mean"])
        varL_array = np.array(data["Energy_EL"]["Variance"])
        
        # 2. Energy VMC (Moyenne par site)
        n = len(e0_array)
        e0_mean = np.mean(e0_array) / L
        var0_mean = np.mean(var0_array) / L
        sem_e0 = np.sqrt(var0_mean) / (np.sqrt(n)) if n > 1 else 0  
        
        # 3. Energy Lanczos (Moyenne par site)
        el_mean = np.mean(el_array) / L
        varL_mean = np.mean(varL_array) / L
        sem_el = np.sqrt(varL_mean) / (np.sqrt(n)) if n > 1 else 0  
        
        # 4. Energy Project avec Zero Variance (Extrapolation linéaire 2 points)
        # Formule : E_proj = E0 - pente * Var0  (où pente = (EL - E0) / (VarL - Var0))
        if var0_mean != varL_mean:
            slope = (el_mean - e0_mean) / (varL_mean - var0_mean)
            e_proj = e0_mean - slope * var0_mean
        else:
            e_proj = e0_mean # Sécurité si la variance est identique
            
        # Ajouter la ligne aux données
        table_data.append({
            "vmc_it": vmc_it,
            "energy VMC": e0_mean,
            "var VMC": var0_mean,
            "sem VMC": sem_e0,
            "energy lanczos": el_mean,
            "var lanczos": varL_mean,
            "sem lanczos": sem_el
        })
        
    # Créer le DataFrame Pandas
    df = pd.DataFrame(table_data)
    
    # Trier par le nombre d'itérations (optionnel mais plus propre)
    try:
        df = df.sort_values(by="vmc_it").reset_index(drop=True)
    except Exception:
        pass # Si vmc_it contient des strings qui ne se trient pas bien, on ignore
        
    return df

import numpy as np

def moyenne_glissante_convolve(data, window_size):
    """
    Lisse un signal 1D en appliquant une moyenne glissante par convolution.
    
    Arguments:
    - data (array/list) : Le tableau de données brutes (ex: energies_mean).
    - window_size (int) : La taille de la fenêtre de lissage (ex: 10 ou 15 points).
    
    Retourne:
    - array : Le tableau lissé. 
              Attention : sa taille sera de (len(data) - window_size + 1).
    """
    if window_size < 2:
        return np.array(data)
    window = np.ones(window_size) / window_size
    donnees_lissees = np.convolve(data, window, mode='valid')
    return donnees_lissees
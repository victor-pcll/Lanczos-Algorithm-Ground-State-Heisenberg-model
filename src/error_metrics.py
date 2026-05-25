import numpy as np
import jax 
import jax.numpy as jnp
from .lanczos_krylov import compute_EL

def compute_EL_with_taylor_naive(E_flat, H2_flat):
    """Naive Taylor propagation assuming strict independence (no covariance)."""
    N = len(E_flat)
    mE = jnp.mean(E_flat)
    err_mE = jnp.std(E_flat, ddof=1) / jnp.sqrt(N)
    
    mH2 = jnp.mean(H2_flat)
    err_mH2 = jnp.std(H2_flat, ddof=1) / jnp.sqrt(N)
    
    eps = E_flat - mE
    nu_vec = H2_flat - 2 * mE * E_flat + mE**2
    eps_nu_vec = eps * nu_vec
    
    M3 = jnp.mean(eps_nu_vec)
    err_M3 = jnp.std(eps_nu_vec, ddof=1) / jnp.sqrt(N)
    
    sig2 = jnp.mean(nu_vec)
    sig = jnp.sqrt(jnp.maximum(sig2, 1e-10))
    mu3 = M3 / (sig**3 + 1e-10)
    
    f_mu = mu3 / 2 - jnp.sqrt(mu3**2 / 4 + 1)
    
    err_sig2 = jnp.sqrt(err_mH2**2 + 4 * mE**2 * err_mE**2)
    err_sig = err_sig2 / (2 * sig)
    err_mu3 = jnp.abs(mu3) * jnp.sqrt((err_M3 / (M3 + 1e-10))**2 + (3 * err_sig / sig)**2)
    
    df_dmu = 0.5 - (mu3 / 4) / jnp.sqrt(mu3**2 / 4 + 1)
    
    err_EL_taylor = jnp.sqrt(
        err_mE**2 + 
        (f_mu * err_sig)**2 + 
        (sig * df_dmu * err_mu3)**2
    )
    return err_EL_taylor

def compute_EL_taylor_blocked(E_2d, H2_2d):
    """Propagation robuste (Matrice de Covariance sur les chaînes). Prend des tableaux 2D."""
    n_chains = E_2d.shape[0]
    mE = jnp.mean(E_2d)
    
    # 1. Calcul instantané pour préserver les corrélations croisées
    eps_2d = E_2d - mE
    nu_2d = H2_2d - 2 * mE * E_2d + mE**2
    eps_nu_2d = eps_2d * nu_2d
    
    # 2. Réduction par blocs (Chaînes) - C'EST ICI LA CLÉ DE LA MÉTHODE !
    E_c = jnp.mean(E_2d, axis=1)
    nu_c = jnp.mean(nu_2d, axis=1)
    eps_nu_c = jnp.mean(eps_nu_2d, axis=1)
    
    sig2 = jnp.mean(nu_c)
    sig = jnp.sqrt(jnp.maximum(sig2, 1e-10))
    M3 = jnp.mean(eps_nu_c)
    mu3 = M3 / (sig**3 + 1e-10)
    
    K = mu3 / 2 - jnp.sqrt(mu3**2 / 4 + 1)
    dK_dmu = 0.5 - (mu3 / 4) / jnp.sqrt(mu3**2 / 4 + 1)
    
    # Matrice de covariance sur les 16 blocs (chaînes)
    data_matrix = jnp.stack([E_c, nu_c, eps_nu_c], axis=0)
    cov_matrix = jnp.cov(data_matrix) / n_chains  # On utilise jnp.cov pour rester dans JAX
    
    var_E0 = cov_matrix[0, 0]
    var_sig2 = cov_matrix[1, 1]
    var_M3 = cov_matrix[2, 2]
    cov_E0_sig2 = cov_matrix[0, 1]
    cov_E0_M3 = cov_matrix[0, 2]
    cov_sig2_M3 = cov_matrix[1, 2]
    
    dsig_dsig2 = 1.0 / (2 * sig)
    var_sig = (dsig_dsig2**2) * var_sig2
    cov_E0_sig = dsig_dsig2 * cov_E0_sig2
    cov_M3_sig = dsig_dsig2 * cov_sig2_M3
    
    dmu_dM3 = 1.0 / (sig**3)
    dmu_dsig = -3.0 * M3 / (sig**4)
    
    var_mu3 = (dmu_dM3**2) * var_M3 + (dmu_dsig**2) * var_sig + 2 * dmu_dM3 * dmu_dsig * cov_M3_sig
    cov_E0_mu3 = dmu_dM3 * cov_E0_M3 + dmu_dsig * cov_E0_sig
    cov_sig_mu3 = dmu_dM3 * cov_M3_sig + dmu_dsig * var_sig
    
    dEL_dE0 = 1.0
    dEL_dsig = K
    dEL_dmu3 = sig * dK_dmu
    
    var_EL = (dEL_dE0**2) * var_E0 + (dEL_dsig**2) * var_sig + (dEL_dmu3**2) * var_mu3 + \
             2 * dEL_dE0 * dEL_dsig * cov_E0_sig + 2 * dEL_dE0 * dEL_dmu3 * cov_E0_mu3 + \
             2 * dEL_dsig * dEL_dmu3 * cov_sig_mu3
             
    return jnp.sqrt(jnp.maximum(var_EL, 0.0))


# --- Jackknife ---
def jackknife_analysis(E_2d, H2_2d):
    """Prend les tableaux 2D, fait le Leave-One-Out sur les chaînes."""
    n_chains = E_2d.shape[0]
    jk_ELs = []
    for i in range(n_chains):
        mask = np.ones(n_chains, dtype=bool)
        mask[i] = False
        E_jk = E_2d[mask].reshape(-1)
        H2_jk = H2_2d[mask].reshape(-1)
        jk_ELs.append(compute_EL(E_jk, H2_jk))

    return np.sqrt((n_chains - 1) * np.var(jk_ELs, ddof=0))

# --- Bootstrap ---
def bootstrap_analysis(E_2d, H2_2d, n_resamples):
    """Prend les tableaux 2D, tire au sort les chaînes entières."""
    n_chains = E_2d.shape[0]
    boot_ELs = []
    for _ in range(n_resamples):
        idx = np.random.randint(0, n_chains, size=n_chains)
        E_resampled = E_2d[idx].reshape(-1)
        H2_resampled = H2_2d[idx].reshape(-1)
        boot_ELs.append(compute_EL(E_resampled, H2_resampled))

    return np.std(boot_ELs)

@jax.jit
def jax_jackknife(E_2d, H2_2d):
    n_chains = E_2d.shape[0]
    
    def single_jk_step(idx_to_drop):
        # 1. On crée un tableau de base de la taille finale voulue (n_chains - 1)
        # Ex: si n_chains=4, base_idx = [0, 1, 2]
        base_idx = jnp.arange(n_chains - 1)
        
        # 2. On décale de +1 tous les indices qui sont >= à l'index à ignorer
        # Ex: si on drop 1, [0, 1, 2] devient [0, 2, 3]
        keep_idx = jnp.where(base_idx >= idx_to_drop, base_idx + 1, base_idx)
        
        # 3. On indexe avec ce tableau de taille 100% garantie statique !
        E_jk = E_2d[keep_idx].reshape(-1)
        H2_jk = H2_2d[keep_idx].reshape(-1)
        
        return compute_EL(E_jk, H2_jk)

    indices = jnp.arange(n_chains)
    jk_ELs = jax.vmap(single_jk_step)(indices)
    
    return jnp.sqrt((n_chains - 1) * jnp.var(jk_ELs, ddof=0))

@jax.jit(static_argnames=['n_resamples'])
def jax_bootstrap(E_2d, H2_2d, n_resamples, key):
    n_chains = E_2d.shape[0]
    
    # 1. On génère TOUS les indices aléatoires d'un coup (Taille : n_resamples x n_chains)
    # C'est immensément plus rapide que de tirer au sort dans une boucle
    idx_matrix = jax.random.randint(key, shape=(n_resamples, n_chains), minval=0, maxval=n_chains)
    
    # 2. On définit ce que fait UN SEUL tirage Bootstrap
    def single_boot_step(indices):
        E_resampled = E_2d[indices].reshape(-1)
        H2_resampled = H2_2d[indices].reshape(-1)
        return compute_EL(E_resampled, H2_resampled)

    # 3. La magie vmap : on applique le calcul sur chaque ligne de idx_matrix
    boot_ELs = jax.vmap(single_boot_step)(idx_matrix)
    
    # 4. Écart-type des estimateurs = erreur standard
    return jnp.std(boot_ELs, ddof=1)
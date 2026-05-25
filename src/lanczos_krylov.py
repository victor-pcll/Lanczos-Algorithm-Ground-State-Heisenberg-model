import jax.numpy as jnp
import jax

@jax.jit
def compute_EL(E_array, H2_array):
    """Calculates Lanczos energy EL from any set of data arrays."""
    mE = jnp.mean(E_array)
    mH2 = jnp.mean(H2_array)
    
    eps = E_array - mE
    nu_vec = H2_array - 2 * mE * E_array + mE**2
    
    sig2 = jnp.mean(nu_vec)
    sig = jnp.sqrt(sig2)
    
    mu3 = (jnp.mean(eps * nu_vec) - jnp.mean(eps) * jnp.mean(nu_vec)) / (sig**3)

    EL = mE + sig * (mu3 / 2 - jnp.sqrt(mu3**2 / 4 + 1))
    return EL

@jax.jit
def compute_Lanczos_energy_and_variance(E_loc, H2_loc):
    """
    Calcule l'énergie de Lanczos ET sa variance en utilisant la base de Krylov 2x2.
    Utilise l'astuce de l'identité pour calculer <H^4> sans opérateur supplémentaire.
    """
    h1 = jnp.mean(E_loc)
    h2 = jnp.mean(H2_loc)
    h3 = jnp.mean(jnp.conj(E_loc) * H2_loc) 
    h4 = jnp.mean(jnp.abs(H2_loc)**2)     
    

    v2 = h2 - h1**2
    v = jnp.sqrt(jnp.maximum(v2, 1e-10))
    
    H_11 = (h3 - 2*h1*h2 + h1**3) / (v2 + 1e-10) + h1
    
    H_mat = jnp.array([
        [h1, v], 
        [v,  H_11]
    ])
    
    eigenvalues, eigenvectors = jnp.linalg.eigh(H_mat)
    eps = E_loc - h1
    nu_vec = H2_loc - 2 * h1 * E_loc + h1**2
    
    sig2 = jnp.mean(nu_vec)
    sig = jnp.sqrt(sig2)
    
    mu3 = (jnp.mean(eps * nu_vec) - jnp.mean(eps) * jnp.mean(nu_vec)) / (sig**3)

    EL = h1 + sig * (mu3 / 2 - jnp.sqrt(mu3**2 / 4 + 1))
    
    u0 = eigenvectors[0, 0]
    u1 = eigenvectors[1, 0]
    
    H2_00 = h2
    H2_01 = (h3 - h1*h2) / (v + 1e-10)
    H2_11 = (h4 - 2*h1*h3 + h1**2 * h2) / (v2 + 1e-10)

    H2_L = (u0**2 * H2_00) + (2 * u0 * u1 * H2_01) + (u1**2 * H2_11)
    
    sigma2_L = H2_L - EL**2
    
    return EL, jnp.maximum(sigma2_L, 0.0)

@jax.jit
def compute_Lanczos_energy_and_variance_robust(E_loc, H2_loc):
    """
    Combine l'algèbre des moments centrés (robustesse absolue) avec 
    l'astuce de l'identité de Luciano pour calculer sigma^2_L.
    """

    mE = jnp.real(jnp.mean(E_loc))
    eps = E_loc - mE
    
    nu_vec = H2_loc - 2 * mE * E_loc + mE**2
    
    sig2 = jnp.real(jnp.mean(nu_vec))                   
    sig = jnp.sqrt(jnp.maximum(sig2, 1e-12))
    
    mu3_unscaled = jnp.real(jnp.mean(jnp.conj(eps) * nu_vec))     # <(H-E0)^3>
    
    mu4_unscaled = jnp.real(jnp.mean(jnp.abs(nu_vec)**2))         # <(H-E0)^4>
    
    mu3 = mu3_unscaled / (sig**3 + 1e-12)
    
    Delta = mu3 / 2.0 - jnp.sqrt(mu3**2 / 4.0 + 1.0)
    
    EL = mE + sig * Delta
    
    # eigenvector components for Lanczos state
    norm = jnp.sqrt(1.0 + Delta**2)
    u0 = 1.0 / norm 
    u1 = Delta / norm
    
    # <H^2> matrix elements in the Krylov basis
    V_00 = sig2
    V_01 = mu3_unscaled / (sig + 1e-12)
    V_11 = mu4_unscaled / (sig2 + 1e-12)

    # Calcul de <H^2> dans l'état de Lanczos
    delta_H2_L = (u0**2 * V_00) + (2 * u0 * u1 * V_01) + (u1**2 * V_11)
    
    sigma2_L = delta_H2_L - (sig * Delta)**2
    
    return EL, jnp.maximum(sigma2_L, 0.0)

@jax.jit
def _compute_all_stats(E_loc_2d, H2_loc_2d):
    """
    Toute la machinerie lourde est compilée en C/XLA ici.
    Aucun objet NetKet, juste des tenseurs.
    """
    E_loc_1d = E_loc_2d.reshape(-1)
    H2_loc_1d = H2_loc_2d.reshape(-1)

    e0 = jnp.mean(E_loc_2d)
    eL, sigma_eL = compute_Lanczos_energy_and_variance_robust(E_loc_1d, H2_loc_1d)
    sigma_e0 = (jnp.mean(H2_loc_1d) - e0**2)
    
    return eL, e0, sigma_e0, sigma_eL


def Krylov_method_with_bootstrap(vstate, ha):
    """
    Fonction appelée dans ta boucle. Gère l'extraction des données puis
    délègue le calcul lourd à la fonction JITée.
    """
    E_loc_2d = vstate.local_estimators(ha).real
    H2_loc_2d = vstate.local_estimators(ha @ ha).real
    
    eL, e0, sigma_e0, sigma_eL = _compute_all_stats(
        E_loc_2d, H2_loc_2d
    )
    
    return (
        eL.item(),
        e0.item(),
        sigma_e0.item(),
        sigma_eL.item()
    )
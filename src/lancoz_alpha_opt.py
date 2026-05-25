import flax.linen as nn
import jax.numpy as jnp 
from typing import Any

class lanczos_ansatz(nn.Module):
    dtype: Any = jnp.complex128 # Défaut utile pour les systèmes quantiques
    alpha_init: complex = 0.01 + 0.0j # On initialise alpha à une valeur proche de 0 (ex: 0.01)

    @nn.compact # permet de def direct dans la partie __call__
    def __call__(self, log_psi_x, eloc_x):
        alpha = self.param('alpha', nn.initializers.constant(self.alpha_init), (1, ), self.dtype)
        return log_psi_x + jnp.log1p(alpha * eloc_x)
    

def _logpsi_lanczos(afun, H, lanczos_variables, x):
    """
    afun: Fonction qui calcule log_psi (ex: lambda x: vstate.log_value(x))
    H: L'Hamiltonien (doit être un JAXOperator, ex: nk.operator.IsingJax)
    lanczos_variables: Le dictionnaire de paramètres de LanczosAnsatz
    x: Batch de configurations (batch_size, N)
    """
    
    logpsi_x = afun(x) # 
    xp, mels = H.get_conn_padded(x)      # xp.shape = (batch_size, max_conn, N), mels.shape = (batch_size, max_conn)
    batch_size, max_conn, N = xp.shape
    xp_flat = xp.reshape(-1, N)
    logpsi_xp_flat = afun(xp_flat)
    logpsi_xp = logpsi_xp_flat.reshape(batch_size, max_conn)
    eloc_x = jnp.sum(mels * jnp.exp(logpsi_xp - logpsi_x[..., None]), axis=-1)

    lanczos_model = lanczos_ansatz(dtype=logpsi_x.dtype)
    out = lanczos_model.apply(lanczos_variables, logpsi_x, eloc_x) 

    return out
    
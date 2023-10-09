import jax.numpy as jnp
from sb3_jax.du.policies import DiffusionBetaScheduler

ddpm_dict = DiffusionBetaScheduler(beta1=1e-4, beta2=0.02, total_denoise_steps=100, method="linear").schedule()

print("sqrt_beta:", ddpm_dict.sqrt_beta_t)

exit()

t = jnp.array([[1],[2],[3]])
sqrtab = ddpm_dict.sqrtab[t]
print(sqrtab.shape)
print(sqrtab)

sqrtab = jnp.repeat(sqrtab, 4, axis=0).reshape(3, -1, 1)
print(sqrtab.shape)
print(sqrtab)

traj = jnp.ones((3, 4, 2))
y_t = traj * sqrtab
print(y_t.shape)
print(y_t)

y_t = y_t.at[:,0,:1].set(jnp.array([[2], [2], [2]]))
print(y_t)

print(ddpm_dict.sqrtmab)
print(ddpm_dict.sqrtab)

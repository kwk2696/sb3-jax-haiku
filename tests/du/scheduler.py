import jax.numpy as jnp
from sb3_jax.common.utils import print_y
from sb3_jax.du.policies import DiffusionBetaScheduler

linear_dict = DiffusionBetaScheduler(beta1=1e-4, beta2=0.02, total_denoise_steps=20, method="linear").schedule()
cosine_dict = DiffusionBetaScheduler(beta1=1e-4, beta2=0.02, total_denoise_steps=20, method="cosine").schedule()

print_y("beta")
print(linear_dict.beta_t)
print(cosine_dict.beta_t)

print_y("bar(a)")
print(linear_dict.alpha_bar_t)
print(cosine_dict.alpha_bar_t)

print_y("bar(a_prev)")
print(linear_dict.alpha_bar_prev_t)
print(cosine_dict.alpha_bar_prev_t)

print_y("1/sqrt(bar(a))")
print(linear_dict.oneover_sqrta)
print(cosine_dict.oneover_sqrta)

print_y("ma_over_sqrtmab_inv")
print(linear_dict.ma_over_sqrtmab_inv)
print(cosine_dict.ma_over_sqrtmab_inv)

print_y("sqrt(b)")
print(linear_dict.sqrt_beta_t)
print(cosine_dict.sqrt_beta_t)

print_y("="*50)
print(linear_dict.alpha_bar_prev_t)
print(cosine_dict.alpha_bar_prev_t)
print(linear_dict.beta_t)
print(cosine_dict.beta_t)
print(linear_dict.posterior_mean_coef1)
print(cosine_dict.posterior_mean_coef1)
print(linear_dict.posterior_mean_coef2)
print(cosine_dict.posterior_mean_coef2)

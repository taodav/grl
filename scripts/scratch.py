from pathlib import Path

from grl.utils.file_system import load_info


#res_path = Path('/home/peter/repos/grl/scripts/runs/obs_dep_uniform_gamma_0.8_0.99_pg_kitchen/parity_check_gvf_obs_rew_seed(2025)_time(20250415-164940)_aed4058b69fe3e5c0bf92fad67876f81.npy')
res_path = Path('/home/peter/repos/grl/results/obs_dep_uniform_gamma_0.8_0.99_rew_in_obs_pg_kitchen/parity_check_gvf_obs_seed(2025)_time(20250415-154949)_b5c7ea5f56ea3595317c14e2cef093d3.npy')
res = load_info(res_path)

print()
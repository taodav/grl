# %% codecell
import numpy as np
import sys
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from argparse import Namespace
from jax.nn import softmax
from jax import config
from pathlib import Path
from collections import namedtuple
from tqdm import tqdm

config.update('jax_platform_name', 'cpu')
np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 24})

from scripts.plotting.parse_experiments import parse_baselines, parse_dirs, parse_batch_dirs
from definitions import ROOT_DIR

colors = {
    'pink': '#ff96b6',
    'red': '#df5b5d',
    'orange': '#DD8453',
    'yellow': '#f8de7c',
    'green': '#3FC57F',
    'cyan': '#48dbe5',
    'blue': '#3180df',
    'purple': '#9d79cf',
    'brown': '#886a2c',
    'white': '#ffffff',
    'light gray': '#d5d5d5',
    'dark gray': '#666666',
    'black': '#000000'
}

belief_perf = {
    '4x3.95': 2.001088974770953,
    'cheese.95': 3.5788796295075453,
    'network': 296.2187032020679,
    'paint.95': 3.293597084371071,
    'parity_check': 0.8099999999999998,
    'shuttle.95': 32.88972468934434,
    'tiger-alt-start': 3.7701893248807115,
    'tmaze_5_two_thirds_up': 2.1257640000000007
}

bars_to_colors = {
    'memoryless': 'light gray',
    'random_2': 'pink',
    'random_4': 'red',
    'ld_2': 'cyan',
    'ld_4': 'blue',
    'sr_discrep_peter_2': 'yellow',
    'sr_discrep_peter_4': 'orange',
}

# %% codecell
# experiment_dirs = [
#     Path(ROOT_DIR, 'results', 'mem_tde_kitchen_sinks_pg'),
#     Path(ROOT_DIR, 'results', 'final_discrep_kitchen_sinks_pg'),
# ]

title = 'SF w/ obs-dependent gamma (U[0.8, 0.99])'
experiment_dirs = [
    Path(ROOT_DIR, 'results', 'dummy_rew_in_obs_pg_kitchen'),

    # Path(ROOT_DIR, 'results', 'ld_pg_kitchen'),
    # Path(ROOT_DIR, 'results', 'gvf_pg_kitchen'),

    # Path(ROOT_DIR, 'results', 'obs_dep_gamma_pg_kitchen'),
    Path(ROOT_DIR, 'results', 'obs_dep_uniform_gamma_0.8_0.99_rew_in_obs_pg_kitchen'),
]

vi_results_dir = Path(ROOT_DIR, 'results', 'vi')
pomdp_files_dir = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files')

args_to_keep = ['spec', 'n_mem_states', 'seed', 'alpha', 'objective', 'gamma_type']
split_by = [arg for arg in args_to_keep if arg != 'seed'] + ['experiment']

# this option allows us to compare to either the optimal belief state soln
# or optimal state soln. ('belief' | 'state')
policy_optim_alg = 'policy_grad'

spec_plot_order = [
    'network',
    'paint.95',
    '4x3.95',
    'tiger-alt-start',
    'shuttle.95',
    'cheese.95',
    'tmaze_5_two_thirds_up',
    'parity_check'
]
obj_to_label = {
    'random': 'Rand',
    'ld': 'LD',
    'sr_discrep_peter': 'GVD'
}

# plot_key = 'final_memoryless_optimal_perf'  # for batch_run
# plot_key = 'final_rand_avg_perf'  # for batch_run

plot_key = 'final_mem_perf'  # for single runs

# %% codecell

compare_to_dict = belief_perf


# %% codecell
# all_res_df = parse_dirs(experiment_dirs,
#                         compare_to_dict,
#                         args_to_keep)
all_res_df = parse_batch_dirs(experiment_dirs,
                              compare_to_dict,
                              args_to_keep)

# peter's inserted data
# 100k iterations, #gammas = 1, min_gamma=0.3,max_gamma=1.0 (fixed)
final_perf_n2_parity_sr = np.array([
    [0.8100258708000183,0.8099725246429443,0.8098602890968323,0.8058860301971436,0.8096286654472351]
])
# 100k iterations, #gammas = 5, min_gamma=0.3  (sweep 3 x 3 = 9)
# commented out row is if you force #gammas=1
final_perf_n4_parity_sr = np.array([
    [0.8102344274520874,0.8100259304046631,0.8100996017456055,0.809973418712616,0.8099787831306458]
    #[0.8099427223205566,0.8100339770317078,0.809939980506897,0.0885670930147171,0.21896842122077945]
])
# 100k iterations, lambdas = {0.0, 0.8}  (sweep 3 x 3 = 9)
final_perf_n2_parity_ld = np.array([
    [0.08340301364660263,0.000001998684183490695,0.4977058470249176,0.31436172127723694,0.16021215915679932,0.00018111758981831372,0.7073317766189575,0.03135672211647034,0.5062191486358643,0.6464985013008118]
])
# 100k iterations, lambdas = {0.0, 0.8}  (sweep 3 x 3 = 9)
final_perf_n4_parity_ld = np.array([
    [0.15158182382583618,0.12360935658216476,0.09604717791080476,0.06442740559577942,0.08323663473129272,0.09481720626354218,0.14268943667411804,0.1438446044921875,0.1181960478425026,0.1142510175704956]
])
# 100k iterations, #gammas = 5, min_gamma=max_gamma = 0.9 (sweep 3 x 3 = 9)
final_perf_n2_tmaze_sr = np.array([
    [1.8765249252319336,1.8627705574035645,1.8416504859924316,1.7613071203231812,1.8218644857406616,1.868623971939087,1.7909445762634275,1.7767953872680664,1.9137930870056152,1.8938632011413576]
])
# 100k iterations, #gammas = 3, min_gamma=max_gamma = 0.9 (sweep 3 x 3 = 9)
final_perf_n4_tmaze_sr = np.array([
    [2.0428552627563477,2.039973735809326,2.043747901916504,2.0228710174560547,2.042267322540283,2.041914463043213]  # cleared 4 nan's  # cleared 4 nan's
])
# 100k iterations, lambdas = {0.2, 1.0}  (sweep 3 x 3 = 9)
final_perf_n2_tmaze_ld = np.array([
    [1.0209728479385376,1.0206257104873655,1.0182665586471558,1.0228257179260254,1.022070288658142,1.0177165269851685,1.017078518867493,1.014500379562378,1.022243857383728,1.0214898586273191]
])
# 100k iterations, lambdas = {0.2, 1.0}  (sweep 3 x 3 = 9)
final_perf_n4_tmaze_ld = np.array([
    [1.4120293855667114,1.6194262504577637,1.5528446435928345,1.5417792797088623,1.41446852684021,0.9921540021896362,1.5397040843963623,1.0114775896072388,1.0336081981658936,1.6207153797149658]
])
#print(all_res_df[all_res_df['objective']!="sr_discrep_peter"]["objective"])
#print(all_res_df[
#    (all_res_df['spec']=="tmaze_5_two_thirds_up")
#    & (all_res_df['objective']=="ld")
#    & (all_res_df['n_mem_states']==2)
#    ])
#sys.exit(0)

update_data_map = {
    ('parity_check', 2, 'sr_discrep_peter'): final_perf_n2_parity_sr,
    ('parity_check', 4, 'sr_discrep_peter'): final_perf_n4_parity_sr,
    ('parity_check', 2, 'ld'): final_perf_n2_parity_ld,
    ('parity_check', 4, 'ld'): final_perf_n4_parity_ld,
    ('tmaze_5_two_thirds_up', 2, 'sr_discrep_peter'): final_perf_n2_tmaze_sr,
    ('tmaze_5_two_thirds_up', 4, 'sr_discrep_peter'): final_perf_n4_tmaze_sr,
    ('tmaze_5_two_thirds_up', 2, 'ld'): final_perf_n2_tmaze_ld,
    ('tmaze_5_two_thirds_up', 4, 'ld'): final_perf_n4_tmaze_ld,
}

# --- 3. Verify the state BEFORE the update for the example case ---
print("--- DataFrame state BEFORE update for the first example ---")
pre_update_rows = all_res_df[
    (all_res_df['spec'] == 'tmaze_5_two_thirds_up') &
    (all_res_df['n_mem_states'] == 2) &
    (all_res_df['objective'] == 'sr_discrep_peter')
]
print(pre_update_rows)#[['spec', 'n_mem_states', 'objective', 'seed', 'final_mem_perf']])
print("\n" + "="*60 + "\n")


# --- 3. Main Update Loop with New Logic ---
print(f"DataFrame initial size: {len(all_res_df)} rows\n")

for (spec, n_mem, objective), new_values in update_data_map.items():
    # Flatten the new values array to 1D
    new_values_flat = new_values.flatten()
    num_new_values = len(new_values_flat)

    # Create a boolean mask to identify the target rows
    base_mask = (
        (all_res_df['spec'] == spec) &
        (all_res_df['n_mem_states'] == n_mem) &
        (all_res_df['objective'] == objective)
    )
    num_rows_to_update = base_mask.sum()

    if num_rows_to_update == 0:
        print(f"!! WARNING: No rows found for ({spec}, {n_mem}, {objective}). Skipping.")
        continue

    # --- NEW LOGIC IS HERE ---
    if num_rows_to_update < num_new_values:
        print(f"Expanding rows for ({spec}, {n_mem}, {objective})...")
        
        # Find the template row (seed == 0)
        template_mask = base_mask & (all_res_df['seed'] == 0)
        template_row_df = all_res_df[template_mask]

        if template_row_df.empty:
            print(f"!! ERROR: Cannot expand. Seed 0 template row not found for ({spec}, {n_mem}, {objective}).")
            continue
        
        # Calculate how many duplicates we need and their new seed values
        num_duplicates_needed = num_new_values - num_rows_to_update
        max_existing_seed = all_res_df[base_mask]['seed'].max()
        new_seeds = range(max_existing_seed + 1, max_existing_seed + 1 + num_duplicates_needed)

        # Create the new rows by duplicating the template
        new_rows_df = pd.concat([template_row_df] * num_duplicates_needed, ignore_index=True)
        new_rows_df['seed'] = list(new_seeds)

        # Append the new rows to the main DataFrame
        all_res_df = pd.concat([all_res_df, new_rows_df], ignore_index=True)
        print(f"  ...appended {num_duplicates_needed} new rows. New total rows: {len(all_res_df)}")
    
    elif num_rows_to_update > num_new_values:
        print(f"!! WARNING: Skipped update for ({spec}, {n_mem}, {objective}). "
              f"Found {num_rows_to_update} rows but data has only {num_new_values} values.")
        continue

    # --- Final Update Step ---
    # This now works for both cases (original size or expanded size)
    # We must re-calculate the mask because the DataFrame has changed
    final_mask = (
        (all_res_df['spec'] == spec) &
        (all_res_df['n_mem_states'] == n_mem) &
        (all_res_df['objective'] == objective)
    )
    
    # Final check before assignment
    if final_mask.sum() == len(new_values_flat):
         all_res_df.loc[final_mask, 'final_mem_perf'] = new_values_flat
         print(f"Successfully updated {final_mask.sum()} rows for ({spec}, {n_mem}, {objective}).\n")
    else:
        print(f"!! ERROR: Mismatch after expansion for ({spec}, {n_mem}, {objective}). "
              f"Found {final_mask.sum()} rows but expected {len(new_values_flat)}.\n")


# --- 4. Verification ---
print("\n" + "="*60 + "\n")
print("--- DataFrame state AFTER update for the first example ---")
print(f"DataFrame final size: {len(all_res_df)} rows")
post_update_rows = all_res_df[
    (all_res_df['spec'] == 'tmaze_5_two_thirds_up') &
    (all_res_df['n_mem_states'] == 2) &
    (all_res_df['objective'] == 'sr_discrep_peter')
]
print(post_update_rows)#[['spec', 'n_mem_states', 'objective', 'seed', 'final_mem_perf']])


# %% codecell

# FILTER OUT for what we want to plot
# alpha = 1.
#
# all_res_df
residual = False
alpha = 1.
filtered_df = all_res_df[
# (all_res_df['residual'] == residual) &
((all_res_df['gamma_type'] == 'fixed') | (all_res_df['gamma_type'] == 'uniform')) &
(all_res_df['alpha'] == alpha)
].reset_index()


# filtered_df['experiment'] = filtered_df['experiment'] + '_' + filtered_df['experiment']


# %% codecell
all_res_groups = filtered_df.groupby(split_by, as_index=False)
all_res_means = all_res_groups.mean()
del all_res_means['seed']
# all_res_means.to_csv(Path(ROOT_DIR, 'results', 'all_pomdps_means.csv'))

# %% codecell
cols_to_normalize = ['init_improvement_perf', plot_key]
# merged_df = filtered_df.merge(compare_to_df, on='spec')
merged_df = filtered_df

# for col_name in cols_to_normalize:

normalized_df = merged_df.copy()
normalized_df['init_improvement_perf'] = (normalized_df['init_improvement_perf'] - merged_df['init_policy_perf']) / (merged_df['compare_to_perf'] - merged_df['init_policy_perf'])
normalized_df[plot_key] = (normalized_df[plot_key] - merged_df['init_policy_perf']) / (merged_df['compare_to_perf'] - merged_df['init_policy_perf'])
del normalized_df['init_policy_perf']
del normalized_df['compare_to_perf']

# %% codecell
normalized_df.loc[(normalized_df['spec'] == 'hallway') & (normalized_df['n_mem_states'] == 8), plot_key] = 0

# %% codecell
# normalized_df[normalized_df['spec'] == 'prisoners_dilemma_all_c']
seeds = normalized_df[normalized_df['spec'] == normalized_df['spec'][0]]['seed'].unique()
# %% codecell
def maybe_spec_map(id: str):
    spec_map = {
        '4x3.95': '4x3',
        'cheese.95': 'cheese',
        'paint.95': 'paint',
        'shuttle.95': 'shuttle',
        'example_7': 'ex. 7',
        'tmaze_5_two_thirds_up': 'tmaze',
        'tiger-alt-start': 'tiger',
        'parity_check': 'parity'
    }

#     spec_map |= prisoners_spec_map

    if id not in spec_map:
        return id
    return spec_map[id]

groups = normalized_df.groupby(split_by, as_index=False)
all_means = groups.mean()
all_means['init_improvement_perf'].clip(lower=0, upper=1, inplace=True)
all_means[plot_key].clip(lower=0, upper=1, inplace=True)

all_std_errs = groups.std()
all_std_errs['init_improvement_perf'] /= np.sqrt(len(seeds))
all_std_errs[plot_key] /= np.sqrt(len(seeds))

# %%

# SORTING
sorted_mean_df = pd.DataFrame()
sorted_std_err_df = pd.DataFrame()

for spec in spec_plot_order:
    mean_spec_df = all_means[all_means['spec'] == spec]
    std_err_spec_df = all_std_errs[all_std_errs['spec'] == spec]
    sorted_mean_df = pd.concat([sorted_mean_df, mean_spec_df])
    sorted_std_err_df = pd.concat([sorted_std_err_df, std_err_spec_df])

# %%
experiments = sorted(normalized_df['experiment'].unique())
objectives = normalized_df['objective'].unique()


group_width = 1
num_n_mem = list(sorted(normalized_df['n_mem_states'].unique()))
n_mem = len(num_n_mem)
n_exp = len(experiments)

specs = sorted_mean_df['spec'].unique()

spec_order_mapping = np.arange(len(specs), dtype=int)


# this was calculated so that there's bar_width / 2 space between subgroups
bar_width = group_width / (1 + (n_mem + 0.5) * n_exp + 1.5)
between_groups_width = bar_width / 2


fig, ax = plt.subplots(figsize=(20, 8))

xlabels = [maybe_spec_map(l) for l in specs]
x = np.arange(len(specs))

example_means = all_means[
                          (all_means['n_mem_states'] == num_n_mem[0]) &
                          (all_means['experiment'] == experiments[0]) &
                          (all_means['objective'] == objectives[0])
                          ]
init_improvement_perf_mean = np.array([example_means[example_means['spec'] == spec]['init_improvement_perf'].item() for spec in specs])
example_std = all_std_errs[
                           (all_std_errs['n_mem_states'] == num_n_mem[0]) &
                           (all_std_errs['experiment'] == experiments[0]) &
                           (all_std_errs['objective'] == objectives[0])
                           ]
init_improvement_perf_std = np.array([example_std[example_std['spec'] == spec]['init_improvement_perf'].item() for spec in specs])

ax.bar(x,
       init_improvement_perf_mean,
       bar_width,
       yerr=init_improvement_perf_std,
       label='Memoryless',
       color=colors[bars_to_colors['memoryless']])

mem_colors = ['#E0B625', '#DD8453', '#C44E52']
exp_hatches = ['/', 'o', '+', '.']
objectives = []

for i, exp_name in enumerate(experiments):
    means = sorted_mean_df[sorted_mean_df['experiment'] == exp_name]
    std_errs = sorted_std_err_df[sorted_std_err_df['experiment'] == exp_name]

    objs = means['objective'].unique()
    assert len(objs) == 1
    objective = objs[0]
    if objective == 'dummy':
        objective = 'random'
    objectives.append(objective)
    # means = sorted_mean_df
    # std_errs = sorted_std_err_df

    for j, n_mem_states in enumerate(num_n_mem):
        curr_mem_mean = np.array(means[means['n_mem_states'] == n_mem_states][plot_key])
        curr_mem_std = np.array(std_errs[std_errs['n_mem_states'] == n_mem_states][plot_key])

        to_add = bar_width + (i + 1) * between_groups_width + (n_mem * bar_width) * i + bar_width * j
        ax.bar(x + to_add,
               curr_mem_mean,
               bar_width,
               yerr=curr_mem_std,
               label=f"{int(np.log2(n_mem_states))} {obj_to_label[objective]} Bit(s)",
               # hatch=exp_hatches[i],
               color=colors[bars_to_colors[f'{objective}_{n_mem_states}']])

ax.set_ylim([0, 1.05])
ax.set_ylabel(f'Relative Performance\n (w.r.t. optimal belief & initial policy)')
ax.set_xticks(x + group_width / 2)
ax.set_xticklabels(xlabels)
#ax.legend(bbox_to_anchor=(0.725, 0.45))
ax.legend(loc="lower left")
# ax.set_title(f"Memory Iteration ({policy_optim_alg})")
# alpha_str = 'uniform' if alpha == 1. else 'occupancy'
residual_str = 'semi_grad' if not residual else 'residual'
title_str = " vs. ".join([f"{obj} ({hatch})" for obj, hatch in zip(objectives, exp_hatches)])
title_str = title + ' ' + title_str
# ax.set_title(f"Memory: (MSTDE (dashes, {residual_str}) vs LD (dots))")
# ax.set_title(title_str)
fig.tight_layout()

plt.show()
#downloads = Path().home() / 'Downloads'
#fig_path = downloads / f"gd_analytical_res.pdf"
fig_path = Path(ROOT_DIR, "plots", f"gd_analytical_res.pdf")
fig.savefig(fig_path, bbox_inches='tight')
# %% codecell
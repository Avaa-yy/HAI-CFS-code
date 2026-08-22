#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plotting.violin_plot import plot_violin
from plotting.bar_plot import plot_bar
from plotting.line_plot import plot_line
from plotting.horizontal_bar_plot import plot_bar_horizontal
from scripts.split_utils import random_train_test_split_indices

DATA_DIR = PROJECT_ROOT / 'data'
FILES_DIR = PROJECT_ROOT / 'files'
FIGURES_DIR = PROJECT_ROOT / 'figures'
FILES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Configure plot appearance.
d_col = ['#2E54A1', '#8B6F47', '#BA3E45']
s_col = ['#81B1D6', '#F4A15D', '#F19CBB']
error_color = '#AFABAB'
human_color = '#2E54A1'
ai_color = '#588E31'


def identity_slug(identity):
    """Return a filesystem-safe English label for participant identity."""
    return {
        'medical_student': 'medical_student',
        'ophthalmologist': 'ophthalmologist',
    }.get(identity, str(identity).strip().lower().replace(' ', '_'))


# Calculate the requested metrics.

def compute_icc2_from_subject_sample_matrix(data_matrix, nan_policy='drop'):
    """Perform the documented analysis step."""
    X = pd.DataFrame(data_matrix).copy()
    X = X.apply(pd.to_numeric, errors='coerce')

    n_subjects_raw, n_samples_raw = X.shape

    if nan_policy == 'drop':
        X = X.dropna(axis=0, how='any')
        X = X.dropna(axis=1, how='any')
    else:
        if X.isna().any().any():
            raise ValueError("data_matrix contains NaN values. The standard ICC(2) ANOVA formula requires a complete matrix; use nan_policy='drop'.")

    X = X.to_numpy(dtype=float)

    n, k = X.shape

    if n < 2:
        raise ValueError(f"Too few valid participants: n={n}")
    if k < 2:
        raise ValueError(f"Too few valid samples: k={k}")

    grand_mean = np.mean(X)
    subject_means = np.mean(X, axis=1)
    sample_means = np.mean(X, axis=0)

    # Aggregate participant-level results.
    ss_subject = k * np.sum((subject_means - grand_mean) ** 2)
    msr = ss_subject / (n - 1)

    # Analysis step.
    ss_sample = n * np.sum((sample_means - grand_mean) ** 2)
    msc = ss_sample / (k - 1)

    # Analysis step.
    residual = X - subject_means[:, None] - sample_means[None, :] + grand_mean
    ss_error = np.sum(residual ** 2)
    mse = ss_error / ((n - 1) * (k - 1))

    # ICC(2,1): two-way random, absolute agreement, single measurement
    icc_2_1 = (msr - mse) / (
        msr + (k - 1) * mse + (k / n) * (msc - mse)
    )

    # Aggregate participant-level results.
    # Analysis step.
    icc_2_k = (msr - mse) / (
        msr + (msc - mse) / n
    )

    return {
        'n_subjects_raw': n_subjects_raw,
        'n_samples_raw': n_samples_raw,
        'n_subjects_used': n,
        'n_samples_used': k,
        'MSR_subject': msr,
        'MSC_sample': msc,
        'MSE_error': mse,
        'ICC_2_1': icc_2_1,
        'ICC_2_k': icc_2_k
    }


def calculate_icc2_for_condition_from_seed_files(
    identi: str = 'ophthalmologist',
    group_id: int = 3,
    n_seeds: int = 100,
    base_seed: int = 42,
    files_dir=FILES_DIR,
    save_dir=FILES_DIR,
    strategy_cols=None,
    save_matrix: bool = False
):
    """Perform the documented analysis step."""
    os.makedirs(save_dir, exist_ok=True)

    if strategy_cols is None:
        strategy_cols = ['Bayesian', 'HAI_CFSd', 'HAI_CFSi', 'Human_Led', 'Human_Only']

    all_subject_runs = []
    loaded_seeds = []

    for i in range(n_seeds):
        seed = base_seed + i

        no_reint_acc = os.path.join(
            files_dir,
            f'aifirst_delegation_accuracy_seed_{seed:03d}.xlsx'
        )

        integration_acc = os.path.join(
            files_dir,
            f'aifirst_integration_accuracy_seed_{seed:03d}.xlsx'
        )

        if not os.path.exists(no_reint_acc) or not os.path.exists(integration_acc):
            # Analysis step.
            continue

        one_seed_df = _extract_subject_synergy_one_seed(
            no_reint_acc_file=no_reint_acc,
            integration_acc_file=integration_acc,
            identi=identi,
            group_id=group_id,
            seed=seed
        )

        all_subject_runs.append(one_seed_df)
        loaded_seeds.append(seed)

    if len(all_subject_runs) == 0:
        raise FileNotFoundError("No seed workbooks were loaded. Check files_dir, base_seed, and n_seeds.")

    all_runs_df = pd.concat(all_subject_runs, ignore_index=True)

    icc_rows = []
    matrices = {}

    safe_identi_name = 'Medical_student' if identi == 'medical_student' else 'Ophthalmologist'

    for strat in strategy_cols:
        # Analysis step.
        mat = (
            all_runs_df
            .pivot_table(
                index='participant_id',
                columns='seed',
                values=strat,
                aggfunc='mean'
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )

        matrices[strat] = mat

        if save_matrix:
            matrix_path = os.path.join(
                save_dir,
                f'icc_matrix_{safe_identi_name}_group{group_id}_{strat}_{n_seeds}seeds.xlsx'
            )
            mat.to_excel(matrix_path)
            # Save the generated result.

        icc_res = compute_icc2_from_subject_sample_matrix(
            mat,
            nan_policy='drop'
        )

        row = {
            'participant_group': identi,
            'confidence_condition': group_id,
            'strategy': strat,
            'loaded_seed_count': len(loaded_seeds),
            **icc_res
        }

        # Analysis step.
        row['ICC_2_100'] = row['ICC_2_k'] if row['n_samples_used'] == 100 else np.nan

        icc_rows.append(row)

    icc_df = pd.DataFrame(icc_rows)

    # Analysis step.
    # print(
    #     icc_df[
    #         [
    #             'participant_group',
    #             'confidence_condition',
    #             'strategy',
    #             'n_subjects_used',
    #             'n_samples_used',
    #             'ICC_2_1',
    #             'ICC_2_k',
    #             'ICC_2_100'
    #         ]
    #     ]
    # )

    return icc_df, matrices


def calculate_icc2_for_all_conditions_from_seed_files(
    n_seeds: int = 100,
    base_seed: int = 42,
    files_dir=FILES_DIR,
    save_dir=FILES_DIR,
    strategy_cols=None,
    save_matrix: bool = False
):
    """Perform the documented analysis step."""
    conditions = [
        ('medical_student', 2),
        ('medical_student', 3),
        ('ophthalmologist', 2),
        ('ophthalmologist', 3)
    ]

    all_icc = []
    all_matrices = {}

    for identi, group_id in conditions:
        # print(f"\n\n==============================")
        # Calculate the requested metrics.
        # print(f"==============================")

        icc_df, matrices = calculate_icc2_for_condition_from_seed_files(
            identi=identi,
            group_id=group_id,
            n_seeds=n_seeds,
            base_seed=base_seed,
            files_dir=files_dir,
            save_dir=save_dir,
            strategy_cols=strategy_cols,
            save_matrix=save_matrix
        )

        all_icc.append(icc_df)
        all_matrices[(identi, group_id)] = matrices

    all_icc_df = pd.concat(all_icc, ignore_index=True)

    return all_icc_df, all_matrices


def plot_icc_hai_cfsd_cfsi_bar(
    icc_file=FILES_DIR / 'icc2_summary_all_conditions_100_seeds.xlsx',
    save_dir=FIGURES_DIR,
    icc_type='2,1',
    save_name=None
):
    """Perform the documented analysis step."""

    os.makedirs(save_dir, exist_ok=True)

    # Load the required data.
    df = pd.read_excel(icc_file)
    df.columns = df.columns.str.strip()

    required_cols = ['participant_group', 'confidence_condition', 'strategy']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"The ICC workbook is missing the required column: {col}")

    # Analysis step.
    if icc_type == '2,1':
        icc_col = 'ICC_2_1'
        y_label = 'ICC(2,1)'
        title = 'ICC(2,1) of HAI-CFSd and HAI-CFSi across four groups'

    elif icc_type == '2,100':
        # Analysis step.
        if 'ICC_2_100' in df.columns and df['ICC_2_100'].notna().any():
            icc_col = 'ICC_2_100'
        # Analysis step.
        elif 'ICC_2_k' in df.columns:
            icc_col = 'ICC_2_k'
            # Analysis step.
        else:
            raise ValueError("The ICC workbook contains neither ICC_2_100 nor ICC_2_k.")

        y_label = 'ICC(2,100)'
        title = 'ICC(2,100) of HAI-CFSd and HAI-CFSi across four groups'

    else:
        raise ValueError("icc_type must be either '2,1' or '2,100'.")

    if icc_col not in df.columns:
        raise ValueError(f"The ICC workbook is missing the required column: {icc_col}")

    # Analysis step.
    target_strategies = ['HAI_CFSd', 'HAI_CFSi']

    plot_df = df[df['strategy'].isin(target_strategies)].copy()

    plot_df['condition'] = (
        plot_df['participant_group'].astype(str) +
        '-condition-' +
        plot_df['confidence_condition'].astype(str)
    )

    # Analysis step.
    plot_wide = plot_df.pivot_table(
        index='condition',
        columns='strategy',
        values=icc_col,
        aggfunc='first'
    )

    condition_order = [
        'medical_student-condition-2',
        'medical_student-condition-3',
        'ophthalmologist-condition-2',
        'ophthalmologist-condition-3'
    ]

    plot_wide = plot_wide.reindex(condition_order)

    # Analysis step.
    # Create the analysis plot.
    # print(plot_wide)

    # Create the analysis plot.
    x = np.arange(len(plot_wide.index))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(
        x - width / 2,
        plot_wide['HAI_CFSd'],
        width,
        label='HAI-CFSd'
    )

    bars2 = ax.bar(
        x + width / 2,
        plot_wide['HAI_CFSi'],
        width,
        label='HAI-CFSi'
    )

    # Analysis step.
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.015,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Configure plot appearance.
    ax.set_xticks(x)
    ax.set_xticklabels(plot_wide.index, fontsize=11)

    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=13)

    ax.set_ylim(0, 1.05)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(frameon=False, fontsize=11)

    plt.tight_layout()

    # Save the generated result.
    if save_name is None:
        safe_icc_name = icc_type.replace(',', '_')
        save_name = f'ICC{safe_icc_name}_HAI_CFSd_CFSi_bar.svg'

    save_path = os.path.join(save_dir, save_name)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        transparent=True
    )

    plt.show()

    # Create the analysis plot.

    return plot_wide


def remove_outliers_iqr(df, column, factor=1.5):
    """Perform the documented analysis step."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Apply the configured random split.
def run_aifirst_processing(input_file: str, output_file: str,
                           fallback_acc_col: str, fallback_conf_col: str,
                           fit_ratio: float = 0.4, random_state: int = 42):
    """Perform the documented analysis step."""
    df = pd.read_excel(input_file)
    df.columns = df.columns.str.strip()
    # Analysis step.
 
    mask_inconsistent = (
        (df['ai_initial_agreement'] == 0) &
        (df['confidence_condition'].isin([2, 3])) &
        ~((df['ai_correct'] == 0) & (df['initial_correct'] == 0))
    )
 
    df['fitted_threshold'] = np.nan
    df['hai_cfs_correct'] = df[fallback_acc_col].copy()
    df['hai_cfs_confidence'] = df[fallback_conf_col].copy()
    df['is_hai_cfs_test'] = 0

    # Apply the configured random split.
 
    inconsistent_df = df[mask_inconsistent].copy()
    if len(inconsistent_df) < 5:
        # Analysis step.
        df.to_excel(output_file, index=False)
        return df
 
    threshold_candidates = np.arange(0.0, 1.01, 0.01)
    rng = np.random.default_rng(random_state)
    grouped = inconsistent_df.groupby(['participant_group', 'confidence_condition'])
 
    for (identity, exp_group), subgroup in grouped:
        if len(subgroup) < 5 or identity == 1:
            # Analysis step.
            continue
 
        # Apply the configured random split.
        original_indices = subgroup.index.values
        try:
            train_indices, test_indices = random_train_test_split_indices(
                original_indices,
                fit_ratio,
                rng,
            )
        except ValueError:
            continue
 
        train_df_tmp = df.loc[train_indices]
        test_df_tmp = df.loc[test_indices]
        # Apply the configured random split.
        #       f"n_train={len(train_indices)}, n_test={len(test_indices)}")
 
        best_threshold = 0.5
        best_accuracy = 0.0
        for thresh in threshold_candidates:
            pred = np.where(train_df_tmp['ai_confidence'] >= thresh,
                            train_df_tmp['ai_correct'],
                            train_df_tmp[fallback_acc_col])
            acc = np.mean(pred)
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = thresh
 
        # Fit or apply the confidence threshold.
              # Analysis step.
 
        this_mask = (df['participant_group'] == identity) & (df['confidence_condition'] == exp_group) & mask_inconsistent
        df.loc[this_mask, 'fitted_threshold'] = best_threshold
 
        test_mask = df.index.isin(test_indices)
        full_test_mask = this_mask & test_mask
        if full_test_mask.sum() > 0:
            df.loc[full_test_mask, 'hai_cfs_correct'] = np.where(
                df.loc[full_test_mask, 'ai_confidence'] >= best_threshold,
                df.loc[full_test_mask, 'ai_correct'],
                df.loc[full_test_mask, fallback_acc_col]
            )
            df.loc[full_test_mask, 'hai_cfs_confidence'] = np.where(
                df.loc[full_test_mask, 'ai_confidence'] >= best_threshold,
                df.loc[full_test_mask, 'ai_confidence'],
                df.loc[full_test_mask, fallback_conf_col]
            )
            df.loc[full_test_mask, 'is_hai_cfs_test'] = 1
 
    df.to_excel(output_file, index=False)
    # Save the generated result.
    return df

# Calculate the requested metrics.
def calculate_incon_accuracies(input_file: str, output_file: str, calculate_pai_ratio: bool = False):
    df0 = pd.read_excel(input_file) #if calculate_pai_ratio else None
    # df0=df0[df0['is_hai_cfs_test'] == 1]
    df = pd.read_excel(input_file)
    df = df[(df['ai_initial_agreement'] == 0) & ~((df['ai_correct'] == 0) & (df['initial_correct'] == 0))]
 
    results = []
    for id_val in range(1, 50):
        for experiment in [1, 2, 3]:
            filtered_df = df[(df["participant_id"] == id_val) & (df["confidence_condition"] == experiment)]
            if filtered_df.empty:
                continue
            if experiment != 1 and 'is_hai_cfs_test' in filtered_df.columns:
                test_df = filtered_df.copy()[filtered_df['is_hai_cfs_test'] == 1]
            else:
                test_df = filtered_df.copy()
            if len(test_df) == 0:
                continue
            ai_accuracy = test_df["ai_correct"].mean()
            initial_accuracy = test_df["initial_correct"].mean()
            final_accuracy = test_df["final_correct"].mean()
            baye_accuracy = -1
            aifirst_accuracy = -1
            aifirst_auc = -1
            AI_auc = -1
            pai_AH_ratio = -1
            if experiment != 1:
                baye_accuracy = test_df["bayesian_correct"].mean()
                aifirst_accuracy = test_df["hai_cfs_correct"].mean()
                y_true0 = test_df["hai_cfs_correct"]
                if len(np.unique(y_true0)) > 1:
                    aifirst_auc = roc_auc_score(y_true0, test_df["hai_cfs_confidence"])
                else:
                    aifirst_auc = np.nan
                if calculate_pai_ratio and df0 is not None:
                    filtered_df0 = df0[(df0["participant_id"] == id_val) & (df0["confidence_condition"] == experiment) & (df0['is_hai_cfs_test'] == 1)]
                    y_true = test_df["ai_correct"].values
                    if len(np.unique(y_true)) > 1:
                        AI_auc = roc_auc_score(y_true, test_df["ai_confidence"])
                    else:
                        AI_auc = np.nan
                    
                    only_conflict=filtered_df.copy()
                    all_tri=df0[(df0["participant_id"] == id_val) & (df0["confidence_condition"] == experiment) ]
                    # # print(len(only_conflict)  ,len(all_tri))
                    pai_AH_ratio = len(only_conflict) / len(all_tri) if len(all_tri) > 0 else -1
            row = [id_val, filtered_df['participant_group'].iloc[0], experiment,
                   ai_accuracy, initial_accuracy, final_accuracy,
                   baye_accuracy, aifirst_accuracy, aifirst_auc]
            if calculate_pai_ratio:
                row += [AI_auc, pai_AH_ratio]
            results.append(row)
 
    if calculate_pai_ratio:
        columns = ["participant_id", 'participant_group',"confidence_condition", "ai_correct", "initial_correct", "final_correct",
                   'bayesian_correct','hai_cfs_correct','hai_cfs_auc','AI_auc','pai_AH_ratio']
    else:
        columns = ["participant_id", 'participant_group',"confidence_condition", "ai_correct", "initial_correct", "final_correct",
                   'bayesian_correct','hai_cfs_correct','hai_cfs_auc']
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_excel(output_file, index=False)
    # Calculate the requested metrics.

# Analysis step.
def extract_condition_results(no_reint_acc_file: str, integration_acc_file: str,
                              identi: str, group_id: int):
    """Perform the documented analysis step."""
    df = pd.read_excel(no_reint_acc_file)
    df.columns = df.columns.str.strip()
    df_2inte = pd.read_excel(integration_acc_file)
    df_2inte.columns = df_2inte.columns.str.strip()

    med = df[df['participant_group'] == identi].copy()
    med_2inte = df_2inte[df_2inte['participant_group'] == identi].copy()

    if group_id == 2:
        g_plot = med[med['confidence_condition'] == 2].sort_values('participant_id').reset_index(drop=True)
        g_plot_2inte = med_2inte[med_2inte['confidence_condition'] == 2].sort_values('participant_id').reset_index(drop=True)
    else:
        g_plot = med[med['confidence_condition'] == 3].sort_values('participant_id').reset_index(drop=True)
        g_plot_2inte = med_2inte[med_2inte['confidence_condition'] == 3].sort_values('participant_id').reset_index(drop=True)

    target_len = max(len(g_plot), len(g_plot_2inte))
    g_plot_AI = pad_with_nan(g_plot['ai_correct'], target_len)

    data_9col = np.column_stack([
        pad_with_nan(g_plot['bayesian_correct'], target_len) - g_plot_AI,
        pad_with_nan(g_plot['hai_cfs_correct'], target_len) - g_plot_AI,
        pad_with_nan(g_plot_2inte['hai_cfs_correct'], target_len) - g_plot_AI,
        pad_with_nan(g_plot['final_correct'], target_len) - g_plot_AI,
        pad_with_nan(g_plot['initial_correct'], target_len) - g_plot_AI,
    ])

    mean_diffs = np.nanmean(data_9col, axis=0)

    # p-value (one-sample t-test vs 0)
    def get_p_vs_zero(col_idx):
        d = data_9col[:, col_idx]
        valid = ~np.isnan(d)
        if valid.sum() < 2:
            return np.nan
        return stats.ttest_1samp(d[valid], 0.0)[1]

    p_cfsd = get_p_vs_zero(1)   # HAI-CFSd
    p_cfsi = get_p_vs_zero(2)   # HAI-CFSi

    return mean_diffs, p_cfsd, p_cfsi

# Main analysis workflow.
def run_stability_analysis(
    n_seeds=100,
    base_seed=42,
    input_file=DATA_DIR / 'retinal_diagnosis_trials_bayesian.xlsx'
):
    os.makedirs(FILES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    # Apply the configured random split.

    results = []
    conditions = [('medical_student', 2), ('medical_student', 3), ('ophthalmologist', 2), ('ophthalmologist', 3)]

    for i in range(n_seeds):
        seed = base_seed + i
        # Analysis step.

        # Analysis step.
        integration_trial = FILES_DIR / f'aifirst_integration_trials_seed_{seed:03d}.xlsx'
        no_reint_trial = FILES_DIR / f'aifirst_delegation_trials_seed_{seed:03d}.xlsx'
        integration_acc = FILES_DIR / f'aifirst_integration_accuracy_seed_{seed:03d}.xlsx'
        no_reint_acc = FILES_DIR / f'aifirst_delegation_accuracy_seed_{seed:03d}.xlsx'

        # 1. Integration
        run_aifirst_processing(input_file, integration_trial,
                                fallback_acc_col='final_correct',
                                fallback_conf_col='final_confidence',
                                fit_ratio=0.4, random_state=seed)
        # 2. Delegation
        run_aifirst_processing(input_file, no_reint_trial,
                                fallback_acc_col='initial_correct',
                                fallback_conf_col='initial_confidence',
                                fit_ratio=0.4, random_state=seed)

        # Calculate the requested metrics.
        calculate_incon_accuracies(integration_trial, integration_acc, calculate_pai_ratio=False)
        calculate_incon_accuracies(no_reint_trial, no_reint_acc, calculate_pai_ratio=True)

        # Analysis step.
        for identi, group in conditions:
            mean_diffs, p_d, p_i = extract_condition_results(no_reint_acc, integration_acc, identi, group)
            row = {
                'seed': seed,
                'identi': identi,
                'group': group,
                'Bayesian': mean_diffs[0],
                'HAI_CFSd': mean_diffs[1],
                'HAI_CFSi': mean_diffs[2],
                'Human_Led': mean_diffs[3],
                'Human_Only': mean_diffs[4],
                'p_CFSd': p_d,
                'p_CFSi': p_i
            }
            results.append(row)

    # Create the analysis plot.
    results_df = pd.DataFrame(results)

    # Analysis step.
    plot_stability_boxplots(results_df)
    plot_pvalue_distributions(results_df)
    # Save the generated result.

def plot_group_mean_distributions(results_df, save_dir=FIGURES_DIR): 
    """Perform the documented analysis step."""
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    from matplotlib.lines import Line2D

    os.makedirs(save_dir, exist_ok=True)
    
    conditions = [('medical_student', 2),('medical_student', 3), ('ophthalmologist', 2), ('ophthalmologist', 3)]  # Analysis step.
    strategy_names = ['HAI_CFSd', 'HAI_CFSi']
    strategy_labels = ['HAI_CFSd', 'HAI_CFSi']
    colors = ['#2E54A1', '#588E31', '#F4A15D', '#BA3E45', '#8B6F47']

    def get_significance_stars(p):
        if pd.isna(p):
            return 'ns'
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'

    def permutation_pvalue_one_sample(data, n_resamples=10000):
        data = data[~np.isnan(data)]
        if len(data) < 3:
            return np.nan

        def statistic(x):
            return np.mean(x)

        try:
            res = stats.permutation_test(
                (data,),
                statistic,
                permutation_type='samples',
                alternative='two-sided',
                n_resamples=n_resamples,
                random_state=42
            )
            return res.pvalue
        except Exception:
            return np.nan

    cm_to_inch = 1 / 2.54
    wigh, heig = 7.5, 7

    color_dict = dict(zip(strategy_names, colors[:len(strategy_names)]))
    label_dict = dict(zip(strategy_names, strategy_labels))

    for identi, group in conditions:
        sub = results_df[
            (results_df['identi'] == identi) & 
            (results_df['group'] == group)
        ].copy()

        df_long = pd.melt(
            sub,
            id_vars=['seed'],
            value_vars=strategy_names,
            var_name='Strategy',
            value_name='Mean_Delta'
        )

        fig, ax = plt.subplots(figsize=(cm_to_inch * wigh, cm_to_inch * heig))

        # Analysis step.
        sns.kdeplot(
            data=df_long,
            x='Mean_Delta',
            hue='Strategy',
            hue_order=strategy_names,
            palette=color_dict,
            fill=True,
            linewidth=2,
            alpha=0.35,
            common_norm=False,
            legend=False,
            ax=ax
        )

        # Analysis step.
        ax.axvline(
            0,
            color='black',
            linestyle='--',
            linewidth=1.2,
            alpha=0.9
        )

        # Calculate the requested metrics.
        star_dict = {}
        for strat in strategy_names:
            vals = sub[strat].values
            p_val = permutation_pvalue_one_sample(vals)
            star_dict[strat] = get_significance_stars(p_val)

        # Analysis step.
        legend_handles = [
            Line2D(
                [0], [0],
                color=color_dict[strat],
                linewidth=2,
                label=f"{label_dict[strat]} {star_dict[strat]}"
            )
            for strat in strategy_names
        ]

        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=8.5,
            loc='upper right'
        )

        # Configure plot appearance.
        ax.set_title(f'{identi} - {group} （100 runs）', pad=10, fontsize=11)
        ax.set_xlabel('Synergy effect', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.7)
        ax.spines['bottom'].set_linewidth(0.7)

        ax.tick_params(axis='both', direction='in', length=3, width=0.8)

        # ax.set_xlim(-0.3, 0.3)

        plt.tight_layout()

        filename = f'group_{identity_slug(identi)}_{group}_mean_distribution'
        save_path = os.path.join(save_dir, f'{filename}.svg')

        plt.savefig(
            save_path,
            format='svg',
            bbox_inches='tight',
            dpi=300,
            transparent=True
        )

        # Analysis step.
        # plt.close(fig)

        # Save the generated result.
# def plot_group_mean_distributions(results_df, save_dir='./fig'):
# Analysis step.
#     import os
#     os.makedirs(save_dir, exist_ok=True)
    
#     conditions = [('medical_student', 2)]#, ('medical_student', 3), ('ophthalmologist', 2), ('ophthalmologist', 3)
#     strategy_names = [ 'HAI_CFSd', 'HAI_CFSi']
#     strategy_labels = [ 'HAI_CFSd', 'HAI_CFSi']
#     colors = ['#2E54A1', '#588E31', '#F4A15D', '#BA3E45', '#8B6F47']

#     def get_significance_stars(p):
#         if pd.isna(p): return 'ns'
#         if p < 0.001: return '***'
#         elif p < 0.01: return '**'
#         elif p < 0.05: return '*'
#         else: return 'ns'

#     def permutation_pvalue_one_sample(data, n_resamples=10000):
#         data = data[~np.isnan(data)]
#         if len(data) < 3: return np.nan
#         def statistic(x): return np.mean(x)
#         try:
#             res = stats.permutation_test((data,), statistic, 
#                                          permutation_type='samples',
#                                          alternative='two-sided',
#                                          n_resamples=n_resamples, random_state=42)
#             return res.pvalue
#         except:
#             return np.nan

#     cm_to_inch = 1 / 2.54
#     wigh, heig = 7.5, 7

#     for identi, group in conditions:
#         sub = results_df[(results_df['identi'] == identi) & (results_df['group'] == group)].copy()
#         df_long = pd.melt(sub, id_vars=['seed'], value_vars=strategy_names,
#                           var_name='Strategy', value_name='Mean_Delta')

#         fig, ax = plt.subplots(figsize=(cm_to_inch * wigh, cm_to_inch * heig))

# Analysis step.
#         sns.kdeplot(
#             data=df_long,
#             x='Mean_Delta',
#             hue='Strategy',
#             # legend=True,
#             hue_order=strategy_names,
#             palette=dict(zip(strategy_names, colors)),
#             fill=True,
#             linewidth=2,
#             alpha=0.35,
#             common_norm=False,
#             ax=ax
#         )

# Analysis step.
#         ax.axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.9)

# Calculate the requested metrics.
#         star_dict = {}
#         for strat in strategy_names:
#             vals = sub[strat].values
#             p_val = permutation_pvalue_one_sample(vals)
#             star_dict[strat] = get_significance_stars(p_val)

# Analysis step.
#         handles, labels = ax.get_legend_handles_labels()
#         new_labels = [f"{lab} {star_dict.get(lab, '')}" for lab in labels]
#         # # print(new_labels)
#         ax.legend(handles, new_labels, 
# Analysis step.
#                   fancybox=False,
#                   edgecolor='none',
#                   fontsize=8.5,
#                   loc='upper right')


        
# Configure plot appearance.
#         ax.set_title(f'{identi} - {group} （100 runs）', pad=10, fontsize=11)
#         ax.set_xlabel('Synergy effct', fontsize=10)
#         ax.set_ylabel('Density', fontsize=10)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_linewidth(0.7)
#         ax.spines['right'].set_linewidth(0.7)
#         ax.spines['left'].set_linewidth(0.7)
#         ax.spines['bottom'].set_linewidth(0.7)
#         ax.tick_params(axis='both', direction='in', length=3, width=0.8)
#         # ax.set_xlim(-0.3, 0.3)

#         plt.tight_layout()
#         filename = f'group_{identi}_{group}_mean_dist_permutation'
#         plt.savefig(f'{save_dir}/{filename}.svg', format='svg', bbox_inches='tight', dpi=300, transparent=True)
#         # plt.close(fig)
# Save the generated result.


# Create the analysis plot.
def plot_stability_boxplots(results_df, save_dir=FIGURES_DIR, n_permutations=10000):
    """Perform the documented analysis step."""
    os.makedirs(save_dir, exist_ok=True)
    conditions = [(identi, g) for identi in ['medical_student', 'ophthalmologist'] for g in [2, 3]]
    strategy_names = ['Bayesian', 'HAI_CFSd', 'HAI_CFSi', 'Human_Led', 'Human_Only']
    strategy_labels = ['Bayesian', 'HAI-CFSd', 'HAI-CFSi', 'Human-Led', 'Human-Only']

    def get_significance_stars(p):
        if pd.isna(p):
            return 'ns'
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'

    # Analysis step.
    def permutation_pvalue_one_sample(data, n_resamples=10000):
        """Perform the documented analysis step."""
        if len(data) < 3 or np.all(np.isnan(data)):
            return np.nan
        
        # Analysis step.
        data = data[~np.isnan(data)]
        
        # Add the configured statistical comparison.
        def statistic(x):
            return np.mean(x)
        
        # Analysis step.
        try:
            res = stats.permutation_test(
                (data,), 
                statistic, 
                permutation_type='samples',      # Analysis step.
                alternative='two-sided',         # Add the configured statistical comparison.
                n_resamples=n_resamples,
                random_state=42                  # Analysis step.
            )
            return res.pvalue
        except Exception as e:
            # Analysis step.
            return np.nan

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (identi, group) in enumerate(conditions):
        sub = results_df[(results_df['identi'] == identi) & (results_df['group'] == group)].copy()
        ax = axes[idx]

        df_melt = pd.melt(sub, id_vars=['seed'], value_vars=strategy_names,
                          var_name='Strategy', value_name='Delta_Acc')

        sns.boxplot(data=df_melt, x='Strategy', y='Delta_Acc', ax=ax,
                    palette=['#2E54A1', '#588E31', '#F4A15D', '#BA3E45', '#8B6F47'])
        sns.stripplot(data=df_melt, x='Strategy', y='Delta_Acc', ax=ax,
                      color='black', alpha=0.35, jitter=True, size=3.5)

        # Calculate the requested metrics.
        y_max = ax.get_ylim()[1]
        for j, strat in enumerate(strategy_names):
            vals = sub[strat].values
            m = np.nanmean(vals)
            sem = stats.sem(vals, nan_policy='omit')
            
            # Analysis step.
            p_val = permutation_pvalue_one_sample(vals, n_resamples=n_permutations)
            stars = get_significance_stars(p_val)

            # ax.text(j, y_max * 0.92, f'{m:.3f}±{sem:.3f}',
            #         ha='center', va='top', fontsize=9, fontweight='bold', color='darkred')
            ax.text(j, y_max * 0.82, stars,
                    ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')

        ax.set_title(f'{identi} - condition {group} (n=100 runs)', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('(Strategy - AI_acc)', fontsize=12)
        ax.set_xlabel('')
        ax.set_xticklabels(strategy_labels, rotation=15, ha='right', fontsize=10)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Stability of strategy improvements over AI across 100 random seeds (permutation test)\n'
                 '(boxplot: median; red line: mean; error bars: SEM; stars: permutation-test p-value vs 0)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'stability_boxplot_permutation_test.svg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # Add the configured statistical comparison.
    # plt.close()



# Analysis step.
def plot_pvalue_distributions(results_df, save_dir=FIGURES_DIR):
    os.makedirs(save_dir, exist_ok=True)
    conditions = [(identi, g) for identi in ['medical_student', 'ophthalmologist'] for g in [2, 3]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (identi, group) in enumerate(conditions):
        sub = results_df[(results_df['identi'] == identi) & (results_df['group'] == group)].copy()
        ax = axes[idx]
        
        sns.histplot(sub['p_CFSd'], bins=25, kde=True, color='#588E31', label='HAI-CFSd', ax=ax, alpha=0.7)
        sns.histplot(sub['p_CFSi'], bins=25, kde=True, color='#F4A15D', label='HAI-CFSi', ax=ax, alpha=0.7)
        
        ax.axvline(0.05, color='red', linestyle='--', linewidth=1.5, label='p=0.05')
        ax.set_title(f"{identi} - condition {group} p-value distribution (100 runs)", fontsize=14, fontweight='bold')
        ax.set_xlabel('p-value (vs 0)', fontsize=12)
        ax.set_ylabel('Count / density', fontsize=12)
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pvalue_distributions_100seeds.svg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    # Save the generated result.


# Create the analysis plot.
def _color_config(identi: str, group_id: int):
    if identi == 'medical_student':
        identi_name = 'Student'
        color_list = [s_col[1]] * 5 if group_id == 2 else [s_col[2]] * 5
    else:
        identi_name = 'Doctor'
        color_list = [d_col[1]] * 5 if group_id == 2 else [d_col[2]] * 5
    return identi_name, color_list

def pad_with_nan(series, target_len):
    if len(series) >= target_len:
        return series.iloc[:target_len].values
    pad = np.full(target_len - len(series), np.nan)
    return np.concatenate([series.values, pad])

def _paired_ttest_print(x, y, label: str):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2:
        # print(f"{label}: not enough valid paired observations")
        return np.nan, np.nan, int(keep.sum() - 1)
    t_stat, p_value = stats.ttest_rel(x[keep], y[keep])
    # print(f"{label}: t={t_stat:.4f}, p={p_value:.4g}, df={keep.sum() - 1}")
    return float(t_stat), float(p_value), int(keep.sum() - 1)

def plot_strategy_comparison(no_reint_accuracy_file: str, integration_accuracy_file: str,
                             method_label: str, identi: str = 'medical_student', group_id: int = 2,
                             y_range=(-1, 0.5), save_dir: str = 'fig'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    identi_name, color_list = _color_config(identi, group_id)
    df = pd.read_excel(no_reint_accuracy_file)
    df.columns = df.columns.str.strip()
    df_2inte = pd.read_excel(integration_accuracy_file)
    df_2inte.columns = df_2inte.columns.str.strip()

    return None  # Return the computed result.

def plot_all_strategy_comparisons(no_reint_accuracy_file: str, integration_accuracy_file: str,
                                  method_label: str, y_range=(-1, 0.5), save_dir: str = 'fig'):
    for identi in ['medical_student', 'ophthalmologist']:
        for group_id in [2, 3]:
            plot_strategy_comparison(no_reint_accuracy_file, integration_accuracy_file,
                                     method_label, identi, group_id, y_range, save_dir)
# Load the required data.

def _extract_subject_synergy_one_seed(no_reint_acc_file: str,
                                      integration_acc_file: str,
                                      identi: str,
                                      group_id: int,
                                      seed: int):
    """Perform the documented analysis step."""
    df_d = pd.read_excel(no_reint_acc_file)
    df_i = pd.read_excel(integration_acc_file)

    df_d.columns = df_d.columns.str.strip()
    df_i.columns = df_i.columns.str.strip()

    # Delegation / no reintegration
    g_d = df_d[
        (df_d['participant_group'] == identi) &
        (df_d['confidence_condition'] == group_id)
    ].copy()

    # Integration
    g_i = df_i[
        (df_i['participant_group'] == identi) &
        (df_i['confidence_condition'] == group_id)
    ].copy()

    # Analysis step.
    g_d = g_d.sort_values('participant_id').drop_duplicates(subset=['participant_id'], keep='first')
    g_i = g_i.sort_values('participant_id').drop_duplicates(subset=['participant_id'], keep='first')

    # Analysis step.
    g_i = g_i[['participant_id', 'hai_cfs_correct']].rename(
        columns={'hai_cfs_correct': 'hai_cfsi_correct'}
    )

    # Analysis step.
    g = g_d.merge(g_i, on='participant_id', how='left')

    out = pd.DataFrame({
        'seed': seed,
        'participant_id': g['participant_id'],
        'participant_group': identi,
        'confidence_condition': group_id,

        # Analysis step.
        'Bayesian': g['bayesian_correct'] - g['ai_correct'],
        'HAI_CFSd': g['hai_cfs_correct'] - g['ai_correct'],
        'HAI_CFSi': g['hai_cfsi_correct'] - g['ai_correct'],
        'Human_Led': g['final_correct'] - g['ai_correct'],
        'Human_Only': g['initial_correct'] - g['ai_correct'],

        # Analysis step.
        'AI_acc': g['ai_correct'],
        'Human_acc': g['initial_correct']
    })

    return out


def load_subject_mean_synergy_from_seed_files(identi: str = 'ophthalmologist',
                                              group_id: int = 3,
                                              n_seeds: int = 100,
                                              base_seed: int = 42,
                                              files_dir=FILES_DIR,
                                              min_valid_seeds: int = 1,
                                              save_excel: bool = True):
    """Perform the documented analysis step."""
    import os

    strategy_cols = ['Bayesian', 'HAI_CFSd', 'HAI_CFSi', 'Human_Led', 'Human_Only']
    all_subject_runs = []

    for i in range(n_seeds):
        seed = base_seed + i

        no_reint_acc = os.path.join(
            files_dir,
            f'aifirst_delegation_accuracy_seed_{seed:03d}.xlsx'
        )
        integration_acc = os.path.join(
            files_dir,
            f'aifirst_integration_accuracy_seed_{seed:03d}.xlsx'
        )

        if not os.path.exists(no_reint_acc) or not os.path.exists(integration_acc):
            # Analysis step.
            continue

        one_seed_df = _extract_subject_synergy_one_seed(
            no_reint_acc_file=no_reint_acc,
            integration_acc_file=integration_acc,
            identi=identi,
            group_id=group_id,
            seed=seed
        )
        all_subject_runs.append(one_seed_df)

    if len(all_subject_runs) == 0:
        raise FileNotFoundError("No seed workbooks were loaded. Check files_dir, base_seed, and n_seeds.")

    all_runs_df = pd.concat(all_subject_runs, ignore_index=True)

    # Aggregate participant-level results.
    mean_part = (
        all_runs_df
        .groupby(['participant_id', 'participant_group', 'confidence_condition'], as_index=False)[strategy_cols + ['AI_acc', 'Human_acc']]
        .mean()
    )
    
    
    # # print(result_accuracy.shape)
    # Example usage.
    # nsub_sam=all_runs_df.groupby(['participant_id', 'participant_group', 'confidence_condition'], as_index=False)['HAI_CFSd']
    # # print('nsub_sam',nsub_sam.shape)
    # icc_single, icc_average = compute_icc_oneway(nsub_sam)
    # Analysis step.
    # Aggregate participant-level results.

    # Aggregate participant-level results.
    count_part = (
        all_runs_df
        .groupby(['participant_id', 'participant_group', 'confidence_condition'], as_index=False)[strategy_cols]
        .count()
        .rename(columns={c: f'n_valid_{c}' for c in strategy_cols})
    )

    subject_mean_df = mean_part.merge(
        count_part,
        on=['participant_id', 'participant_group', 'confidence_condition'],
        how='left'
    ).sort_values('participant_id').reset_index(drop=True)

    # Filter the relevant observations.
    if min_valid_seeds is not None and min_valid_seeds > 1:
        valid_cols = [f'n_valid_{c}' for c in strategy_cols]
        before_n = len(subject_mean_df)
        subject_mean_df = subject_mean_df[
            subject_mean_df[valid_cols].min(axis=1) >= min_valid_seeds
        ].copy()
        after_n = len(subject_mean_df)
        # Filter the relevant observations.

    if save_excel:
        identi_name = 'Medical_student' if identi == 'medical_student' else 'Ophthalmologist'
        out_path = os.path.join(
            files_dir,
            f'subject_mean_synergy_{identi_name}_group{group_id}_{n_seeds}seeds.xlsx'
        )
        subject_mean_df.to_excel(out_path, index=False)
        # Aggregate participant-level results.

    return subject_mean_df

def plot_subject_mean_synergy_bar(identi: str = 'ophthalmologist',
                                  group_id: int = 3,
                                  n_seeds: int = 100,
                                  base_seed: int = 42,
                                  files_dir=FILES_DIR,
                                  save_dir=FIGURES_DIR,
                                  y_range=(-1, 0.5),
                                  min_valid_seeds: int = 1,
                                  scatter_plot: bool = True):
    """Perform the documented analysis step."""
    import os
    from scipy import stats

    os.makedirs(save_dir, exist_ok=True)

    strategy_cols = ['Bayesian', 'HAI_CFSd', 'HAI_CFSi', 'Human_Led', 'Human_Only']
    col_names = ['Bayesian', 'HAI-CFSd', 'HAI-CFSi', 'Human-Led', 'Human-Only']

    # Configure plot appearance.
    if identi == 'medical_student':
        identi_name = 'Student'
        color_list = [s_col[1]] * 5 if group_id == 2 else [s_col[2]] * 5
    else:
        identi_name = 'Doctor'
        color_list = [d_col[1]] * 5 if group_id == 2 else [d_col[2]] * 5

    subject_mean_df = load_subject_mean_synergy_from_seed_files(
        identi=identi,
        group_id=group_id,
        n_seeds=n_seeds,
        base_seed=base_seed,
        files_dir=files_dir,
        min_valid_seeds=min_valid_seeds,
        save_excel=True
    )

    # Analysis step.
    # Aggregate participant-level results.
    result_accuracy = subject_mean_df[strategy_cols].to_numpy(dtype=float)

    # Analysis step.
    # Aggregate participant-level results.
    # Analysis step.
    # for name, val in zip(col_names, np.nanmean(result_accuracy, axis=0)):
    #     print(f"  {name}: {val:.4f}")

    # Analysis step.
    # Bayesian vs mean(HAI-CFSd, HAI-CFSi)
    baye = result_accuracy[:, 0]
    hai_mean = np.nanmean(result_accuracy[:, [1, 2]], axis=1)
    valid = np.isfinite(baye) & np.isfinite(hai_mean)

    # if valid.sum() >= 2:
    #     t_stat, p_value = stats.ttest_rel(baye[valid], hai_mean[valid])
    #     # print(
    #         f"t,p,n-1 between Bayesian and mean(HAI-CFSd, HAI-CFSi): "
    #         f"{t_stat:.4f}, {p_value:.4f}, {valid.sum() - 1}"
    #     )
    # else:
    # Aggregate participant-level results.

    # print('AI acc mean across subjects:', np.nanmean(subject_mean_df['AI_acc']))
    # print('Human acc mean across subjects:', np.nanmean(subject_mean_df['Human_acc']))
    # print('HAI_CFSd across subjects:', np.nanmean(subject_mean_df['HAI_CFSd']))
    # print('HAI_CFSi across subjects:', np.nanmean(subject_mean_df['HAI_CFSi']))


    save_path = os.path.join(
        save_dir,
        f'{identi_name} strategy ACC compare {group_id}_subject_mean_{n_seeds}seeds.svg'
    )
    

    
    plot_bar_horizontal(
        data=result_accuracy,
        figsize_cm=(7.5, 6),
        ylim=y_range,
        line_plot=[],
        large_gap_cols=[],
        col_names=col_names,
        ylabel='Accuracy - AI Accuracy',
        title='',
        bar_colors=color_list,
        sig_test_vs_zero=[0, 1, 2, 3, 4],
        paired_tests=[(0, 1), (1, 2)],
        errorbar='sem',
        alpha_scatter=0.8,
        alpha_hist=0.5,
        save_path=save_path,
        scatter_plot=scatter_plot
    )

    # Create the analysis plot.

    return result_accuracy, subject_mean_df


def plot_all_subject_mean_synergy_bars(n_seeds: int = 100,
                                       base_seed: int = 42,
                                       files_dir=FILES_DIR,
                                       save_dir=FIGURES_DIR,
                                       y_range=(-1, 0.5),
                                       min_valid_seeds: int = 1):
    """Perform the documented analysis step."""
    outputs = {}

    for identi in ['medical_student', 'ophthalmologist']:
        for group_id in [2, 3]:
            result_accuracy, subject_mean_df = plot_subject_mean_synergy_bar(
                identi=identi,
                group_id=group_id,
                n_seeds=n_seeds,
                base_seed=base_seed,
                files_dir=files_dir,
                save_dir=save_dir,
                y_range=y_range,
                min_valid_seeds=min_valid_seeds,
                scatter_plot=True
            )
            outputs[(identi, group_id)] = {
                'result_accuracy': result_accuracy,
                'subject_mean_df': subject_mean_df
            }

    return outputs
# Load the required data.

def _extract_subject_scatter_metrics_one_seed(acc_file: str,
                                              seed: int,
                                              group_id: int = 1,
                                              identi=None,
                                              strategy_col: str = 'hai_cfs_correct'):
    """Perform the documented analysis step."""
    df = pd.read_excel(acc_file)
    df.columns = df.columns.str.strip()

    if identi is not None and 'participant_group' in df.columns:
        df = df[df['participant_group'] == identi].copy()

    df = df[df['confidence_condition'] == group_id].copy()
    df = df.sort_values('participant_id').drop_duplicates(subset=['participant_id'], keep='first')

    required_cols = ['participant_id', 'ai_correct', 'initial_correct', strategy_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{acc_file} is missing the required column: {c}")

    if 'AI_auc' not in df.columns:
        raise ValueError(f"{acc_file} is missing AI_auc, which is required for the AI AUC scatter plot.")

    out = pd.DataFrame({
        'seed': seed,
        'participant_id': df['participant_id'],
        'confidence_condition': group_id,
        'synergy_effect': df[strategy_col] - df['ai_correct'],
        'AI_acc': df['ai_correct'],
        'AI_auc': df['AI_auc'],
        'Human_acc': df['initial_correct']
    })

    if 'participant_group' in df.columns:
        out['participant_group'] = df['participant_group'].values

    if 'pai_AH_ratio' in df.columns:
        out['pai_AH_ratio'] = df['pai_AH_ratio']
        # Analysis step.
        # x1 = initial_correct * pai_AH_ratio
        # x2 = ai_correct * pai_AH_ratio
        out['Part_Human_acc'] = df['initial_correct'] * df['pai_AH_ratio']
        out['Part_AI_acc'] = df['ai_correct'] * df['pai_AH_ratio']
    else:
        out['pai_AH_ratio'] = np.nan
        out['Part_Human_acc'] = np.nan
        out['Part_AI_acc'] = np.nan

    return out


def load_subject_mean_scatter_metrics_from_seed_files(file_pattern: str,
                                                      n_seeds: int = 100,
                                                      base_seed: int = 42,
                                                      group_id: int = 1,
                                                      identi=None,
                                                      strategy_col: str = 'hai_cfs_correct',
                                                      save_excel: bool = True,
                                                      save_path: str = None):
    """Perform the documented analysis step."""
    import os

    all_runs = []

    for i in range(n_seeds):
        seed = base_seed + i
        acc_file = file_pattern.format(seed=seed)

        if not os.path.exists(acc_file):
            # Analysis step.
            continue

        one_seed_df = _extract_subject_scatter_metrics_one_seed(
            acc_file=acc_file,
            seed=seed,
            group_id=group_id,
            identi=identi,
            strategy_col=strategy_col
        )
        all_runs.append(one_seed_df)

    if len(all_runs) == 0:
        raise FileNotFoundError("No seed workbooks were loaded. Check file_pattern, base_seed, and n_seeds.")

    all_runs_df = pd.concat(all_runs, ignore_index=True)

    mean_cols = [
        'synergy_effect',
        'AI_acc',
        'AI_auc',
        'Human_acc',
        'pai_AH_ratio',
        'Part_Human_acc',
        'Part_AI_acc'
    ]

    group_cols = ['participant_id', 'confidence_condition']
    if 'participant_group' in all_runs_df.columns:
        group_cols = ['participant_id', 'participant_group', 'confidence_condition']

    subject_mean_df = (
        all_runs_df
        .groupby(group_cols, as_index=False)[mean_cols]
        .mean()
        .sort_values('participant_id')
        .reset_index(drop=True)
    )

    # Aggregate participant-level results.
    n_valid_df = (
        all_runs_df
        .groupby(group_cols, as_index=False)['synergy_effect']
        .count()
        .rename(columns={'synergy_effect': 'n_valid_seeds'})
    )

    subject_mean_df = subject_mean_df.merge(n_valid_df, on=group_cols, how='left')

    # Load the required data.
    # Aggregate participant-level results.
    # Analysis step.
    # for c in ['synergy_effect', 'AI_acc', 'AI_auc', 'Human_acc', 'Part_AI_acc', 'Part_Human_acc']:
    #     if c in subject_mean_df.columns:
    #         print(f"  {c}: {subject_mean_df[c].min():.4f} ~ {subject_mean_df[c].max():.4f}")

    if save_excel:
        if save_path is None:
            save_path = FILES_DIR / f'subject_mean_scatter_metrics_group_{group_id}_{n_seeds}_seeds.xlsx'
        subject_mean_df.to_excel(save_path, index=False)
        # Aggregate participant-level results.

    return subject_mean_df


def _plot_one_mean_scatter(subject_mean_df: pd.DataFrame,
                           x_col: str,
                           y_col: str = 'synergy_effect',
                           x_label: str = '',
                           y_label: str = 'Synergy effect',
                           color='#8B6F47',
                           save_path=FIGURES_DIR / 'scatter.svg',
                           x_ticks=None,
                           y_ticks=None,
                           filter_auc_min: float = 0.4,
                           figsize_cm=(7, 6.6)):
    """Perform the documented analysis step."""
    import os
    from scipy.stats import pearsonr

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_plot = subject_mean_df.copy()

    # Analysis step.
    if filter_auc_min is not None:
        df_plot = df_plot[df_plot['AI_auc'] > filter_auc_min].copy()

    x = df_plot[x_col].to_numpy(dtype=float)
    y = df_plot[y_col].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    cm_to_inch = 1 / 2.54
    plt.figure(figsize=(figsize_cm[0] * cm_to_inch, figsize_cm[1] * cm_to_inch))

    plt.scatter(
        x, y,
        color=color,
        s=55,
        alpha=0.85,
        edgecolors='white',
        linewidth=1
    )

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)

        if x_ticks is not None:
            x_line = np.linspace(x_ticks[0], x_ticks[-1], 100)
        else:
            x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)

        y_line = slope * x_line + intercept

        plt.plot(
            x_line, y_line,
            color=color,
            linewidth=2
        )

        r, p = pearsonr(x, y)
        p_str = f"{p:.2e}" if p < 0.0001 else f"{p:.4f}"

        plt.legend(
            [f'r = {r:.3f}, p = {p_str}'],
            frameon=False,
            loc='best',
            handlelength=0,
            handletextpad=0
        )

        # print(f"{x_label}: n={len(x)}, r={r:.4f}, p={p_str}")
    else:
        print(f"{x_label}: fewer than two valid points; correlation was not calculated")

    if x_ticks is not None:
        plt.gca().set_xticks(x_ticks)

    if y_ticks is not None:
        plt.gca().set_yticks(y_ticks)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

    # Create the analysis plot.


def plot_subject_mean_scatter_from_seed_files(file_pattern: str,
                                              n_seeds: int = 100,
                                              base_seed: int = 42,
                                              group_id: int = 1,
                                              identi=None,
                                              strategy_col: str = 'hai_cfs_correct',
                                              save_dir=FIGURES_DIR,
                                              save_prefix: str = 'radio',
                                              color=None,
                                              use_part_acc: bool = True,
                                              filter_auc_min: float = 0.4):
    """Perform the documented analysis step."""
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Configure plot appearance.
    if color is None:
        try:
            if identi == 'medical_student':
                color = s_col[1] if group_id == 2 else s_col[2]
            else:
                color = d_col[1] if group_id in [1, 2] else d_col[2]
        except NameError:
            color = '#8B6F47'

    metric_save_path = FILES_DIR / f'{save_prefix}_subject_mean_scatter_metrics_group_{group_id}_{n_seeds}_seeds.xlsx'

    subject_mean_df = load_subject_mean_scatter_metrics_from_seed_files(
        file_pattern=file_pattern,
        n_seeds=n_seeds,
        base_seed=base_seed,
        group_id=group_id,
        identi=identi,
        strategy_col=strategy_col,
        save_excel=True,
        save_path=metric_save_path
    )

    y_ticks = np.arange(-0.2,0.41, 0.2)

    # 1. Synergy effect vs AI AUC
    _plot_one_mean_scatter(
        subject_mean_df=subject_mean_df,
        x_col='AI_auc',
        y_col='synergy_effect',
        x_label='AI AUC',
        y_label='Synergy effect',
        color=color,
        save_path=os.path.join(save_dir, f'scatter_synergy_effect_vs_ai_auc_{save_prefix}_{n_seeds}seeds.svg'),
        x_ticks=np.arange(0.5, 1.01, 0.1),
        y_ticks=y_ticks,
        filter_auc_min=filter_auc_min
    )

    # 2. Synergy effect vs AI acc
    if use_part_acc:
        subject_mean_df['AI_acc'] = (
           subject_mean_df['AI_acc'] * subject_mean_df['pai_AH_ratio']
        )
        # # print(subject_mean_df['pai_AH_ratio'])
        subject_mean_df['Human_acc'] = (
           subject_mean_df['Human_acc'] * subject_mean_df['pai_AH_ratio']
        )
        ai_x_col = 'Part_AI_acc'
        ai_x_label = 'Part AI acc'
        ai_x_ticks = np.arange(0, 0.51, 0.1)
        ai_suffix = 'part_ai_acc'
        
        
    else:
        ai_x_col = 'AI_acc'
        ai_x_label = 'AI acc'
        ai_x_ticks = np.arange(0, 1.01, 0.2)
        ai_suffix = 'ai_acc'

    y_ticks = np.arange(-0.2,0.41, 0.2)
    _plot_one_mean_scatter(
        subject_mean_df=subject_mean_df,
        x_col=ai_x_col,
        y_col='synergy_effect',
        x_label=ai_x_label,
        y_label='Synergy effect',
        color=color,
        save_path=os.path.join(save_dir, f'scatter_synergy_effect_vs_{ai_suffix}_{save_prefix}_{n_seeds}seeds.svg'),
        x_ticks=ai_x_ticks,
        y_ticks=y_ticks,
        filter_auc_min=filter_auc_min
    )

    # 3. Synergy effect vs human acc
    if use_part_acc:
        subject_mean_df['AI_acc'] = (
           subject_mean_df['AI_acc'] * subject_mean_df['pai_AH_ratio']
        )
        subject_mean_df['Human_acc'] = (
           subject_mean_df['Human_acc'] * subject_mean_df['pai_AH_ratio']
        )
        human_x_col = 'Part_Human_acc'
        human_x_label = 'Part human acc'
        human_x_ticks = np.arange(0, 0.21, 0.1)
        human_suffix = 'part_human_acc'
    else:
        human_x_col = 'Human_acc'
        human_x_label = 'Human acc'
        human_x_ticks = np.arange(0, 1.01, 0.2)
        human_suffix = 'human_acc'

    y_ticks = np.arange(-0.2,0.41, 0.2)
    _plot_one_mean_scatter(
        subject_mean_df=subject_mean_df,
        x_col=human_x_col,
        y_col='synergy_effect',
        x_label=human_x_label,
        y_label='Synergy effect',
        color=color,
        save_path=os.path.join(save_dir, f'scatter_synergy_effect_vs_{human_suffix}_{save_prefix}_{n_seeds}seeds.svg'),
        x_ticks=human_x_ticks,
        y_ticks=y_ticks,
        filter_auc_min=filter_auc_min
    )

    return subject_mean_df

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def _get_condition_color(identi, group_id):
    """Perform the documented analysis step."""
    try:
        if identi == 'medical_student':
            return s_col[1] if group_id == 2 else s_col[2]
        else:
            return d_col[1] if group_id == 2 else d_col[2]
    except:
        # fallback
        fallback = {
            ('medical_student', 2): '#F4A15D',
            ('medical_student', 3): '#F19CBB',
            ('ophthalmologist', 2): '#8B6F47',
            ('ophthalmologist', 3): '#BA3E45',
        }
        return fallback.get((identi, group_id), '#4C72B0')


def collect_tau_from_seed_files(
    file_pattern,
    n_seeds=100,
    base_seed=42,
    threshold_col='fitted_threshold',
    conditions=None,
    save_excel=True,
    save_path=None
):
    """Perform the documented analysis step."""
    if conditions is None:
        conditions = [('medical_student', 2), ('medical_student', 3), ('ophthalmologist', 2), ('ophthalmologist', 3)]

    rows = []

    for i in range(n_seeds):
        # print(i)
        seed = base_seed + i
        file_path = file_pattern.format(seed=seed)

        if not os.path.exists(file_path):
            # Analysis step.
            continue

        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()

        if threshold_col not in df.columns:
            raise ValueError(f"{file_path} does not contain the column: {threshold_col}")

        for identi, group_id in conditions:
            sub = df[
                (df['participant_group'] == identi) &
                (df['confidence_condition'] == group_id)
            ][threshold_col].dropna()

            # Analysis step.
            sub = sub[np.isfinite(sub)]

            if len(sub) == 0:
                tau_val = np.nan
            else:
                # Fit or apply the confidence threshold.
                # Analysis step.
                unique_vals = np.unique(np.round(sub.astype(float), 10))

                if len(unique_vals) == 1:
                    tau_val = unique_vals[0]
                else:
                    # Fit or apply the confidence threshold.
                    # Analysis step.
                    tau_val = pd.Series(np.round(sub.astype(float), 10)).mode().iloc[0]

            rows.append({
                'seed': seed,
                'identi': identi,
                'group': group_id,
                'tau': tau_val
            })

    tau_df = pd.DataFrame(rows)

    if save_excel:
        if save_path is None:
            safe_name = (
                os.path.basename(file_pattern)
                .replace('{seed:03d}', 'all_seeds')
                .replace('{seed}', 'all_seeds')
                .replace('.xlsx', '')
            )
            save_path = FILES_DIR / f'{safe_name}_threshold_summary.xlsx'
        tau_df.to_excel(save_path, index=False)
        # Save the generated result.

    return tau_df


def plot_tau_distributions(
    tau_df,
    save_dir=FIGURES_DIR,
    save_prefix='aifirst_tau',
    conditions=None
):
    """Perform the documented analysis step."""
    os.makedirs(save_dir, exist_ok=True)

    if conditions is None:
        conditions = [('medical_student', 2), ('medical_student', 3), ('ophthalmologist', 2), ('ophthalmologist', 3)]

    cm_to_inch = 1 / 2.54
    wigh, heig = 7.5, 7

    for identi, group_id in conditions:
        sub = tau_df[
            (tau_df['identi'] == identi) &
            (tau_df['group'] == group_id)
        ].copy()

        sub = sub[np.isfinite(sub['tau'])].copy()

        if len(sub) == 0:
            # Create the analysis plot.
            continue

        color = _get_condition_color(identi, group_id)

        fig, ax = plt.subplots(figsize=(cm_to_inch * wigh, cm_to_inch * heig))

        # Analysis step.
        # ax.hist(
        #     sub['tau'],
        #     bins=np.linspace(0, 0.8, 21),
        #     density=True,
        #     color=color,
        #     alpha=0.18,
        #     edgecolor='none'
        # )

        # Analysis step.
        if len(sub) >= 2 and sub['tau'].nunique() > 1:
            sns.kdeplot(
                data=sub,
                x='tau',
                fill=True,
                linewidth=2,
                alpha=0.35,
                color=color,
                # Analysis step.
                clip=(0, 1),    # Analysis step.
                bw_adjust=2.0,
                legend=False,
                ax=ax
            )

        # # rug
        # sns.rugplot(
        #     data=sub,
        #     x='tau',
        #     color=color,
        #     alpha=0.45,
        #     height=0.05,
        #     ax=ax
        # )

        # Analysis step.
        mean_tau = sub['tau'].mean()
        median_tau = sub['tau'].median()
        sd_tau = sub['tau'].std(ddof=1) if len(sub) > 1 else 0.0

        ax.axvline(mean_tau, color='black', linestyle='--', linewidth=1.5, alpha=0.95)
        # ax.axvline(median_tau, color='black', linestyle=':', linewidth=1.2, alpha=0.85)

        # Analysis step.
        legend_handles = [
            Line2D([0], [0], color='black', lw=2, label=f'{mean_tau:.3f}'),
            # Line2D([0], [0], color='black', lw=1.2, linestyle=':', label=f'Median = {median_tau:.3f}'),
            # Line2D([0], [0], color='none', lw=0, label=f'SD = {sd_tau:.3f}'),
            # Line2D([0], [0], color='none', lw=0, label=f'n = {len(sub)}'),
        ]
        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=8.5,
            loc='upper right'
        )

        # Configure plot appearance.
        ax.set_title(f'{identi} - condition {group_id} (100 runs)', pad=10, fontsize=11)
        ax.set_xlabel(r'Threshold', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)

        ax.set_xlim(0, 0.81)
        ax.set_xticks(np.arange(0, 0.81, 0.2))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.7)
        ax.spines['bottom'].set_linewidth(0.7)

        ax.tick_params(axis='both', direction='in', length=3, width=0.8)

        plt.tight_layout()

        filename = (
            f'{save_prefix}_{identity_slug(identi)}_group_{group_id}_'
            'threshold_distribution.svg'
        )
        save_path = os.path.join(save_dir, filename)

        plt.savefig(
            save_path,
            format='svg',
            bbox_inches='tight',
            dpi=300,
            transparent=True
        )
        # plt.close(fig)

        # Save the generated result.


def load_and_plot_tau_distributions(
    file_pattern,
    n_seeds=100,
    base_seed=42,
    threshold_col='fitted_threshold',
    save_dir=FIGURES_DIR,
    save_prefix='aifirst_tau',
    conditions=None,
    save_excel=True,
    save_excel_path=None
):
    """Perform the documented analysis step."""
    # tau_df = collect_tau_from_seed_files(
    #     file_pattern=file_pattern,
    #     n_seeds=n_seeds,
    #     base_seed=base_seed,
    #     threshold_col=threshold_col,
    #     conditions=conditions,
    #     save_excel=save_excel,
    #     save_path=save_excel_path
    # )

    # Plot or analysis configuration.
    if save_excel_path is None:
        safe_name = (
            os.path.basename(file_pattern)
            .replace('{seed:03d}', 'all_seeds')
            .replace('{seed}', 'all_seeds')
            .replace('.xlsx', '')
        )
        save_excel_path = FILES_DIR / f'{safe_name}_threshold_summary.xlsx'
    
    tau_df = pd.read_excel(save_excel_path)
    # # print(tau_df.shape)
    tau_df.columns = tau_df.columns.str.strip()
    
    
    # Plot or analysis configuration.
    tau_df = (
        tau_df
        .sort_values(['identi', 'group', 'seed'])
        .groupby(['identi', 'group'], as_index=False)
        .head(100)
        .reset_index(drop=True)
    )
    
    # print(tau_df.groupby(['identi', 'group']).size())
    # print(f'load tau in file {save_excel_path}')
    plot_tau_distributions(
        tau_df=tau_df,
        save_dir=save_dir,
        save_prefix=save_prefix,
        conditions=conditions
    )

    return tau_df
# In[]
# Main analysis workflow.
if __name__ == "__main__":
    ## In[0]
    '''sample 100 and save files'''
    run_stability_analysis(
        n_seeds=100,
        base_seed=42,
        input_file=DATA_DIR / 'retinal_diagnosis_trials_bayesian.xlsx'
    )
    # Analysis step.
    
    # In[1]
    """Perform the documented analysis step."""
    result_accuracy, subject_mean_df = plot_subject_mean_synergy_bar(
        identi='ophthalmologist',
        group_id=2,
        n_seeds=100,
        base_seed=42,
        files_dir=FILES_DIR,
        save_dir=FIGURES_DIR,
        y_range=(-1, 0.5),
        min_valid_seeds=1
    )
    # In[2]
    '''plot correlation between synergy effct and AI AUC/AI acc/Human acc'''
    # iden='medical_student'
    iden='ophthalmologist'
    # grou_id=2
    grou_id=3
    subject_mean_df = plot_subject_mean_scatter_from_seed_files(
        file_pattern=str(FILES_DIR / 'aifirst_delegation_accuracy_seed_{seed:03d}.xlsx'),
        n_seeds=100,
        base_seed=42,
        group_id=grou_id,
        identi=iden,
        strategy_col='hai_cfs_correct',
        save_dir=FIGURES_DIR,
        save_prefix=f'retina_ophthalmologist_group_{grou_id}',
        use_part_acc=True,
        filter_auc_min=0.4
    )

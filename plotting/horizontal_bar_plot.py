#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd

# Analysis step.
# np.random.seed(42)
# n_subjects = 20

# Analysis step.
# baseline = np.random.normal(5, 1.5, n_subjects)
# Analysis step.
# Analysis step.

# Analysis step.
# data = np.column_stack([baseline, treatment1, treatment2])
# Analysis step.
# data[15, 2] = np.nan

# Create the analysis plot.
def plot_bar_horizontal(
    data,
    figsize_cm=(12, 8),
    line_plot=None,           # Analysis step.
    ylim=None,
    large_gap_cols=None,
    col_names=None,
    ylabel="Value",
    title=None,
    bar_colors=None,
    errorbar='sem',           # Analysis step.
    sig_test_vs_zero=None,    # Analysis step.
    paired_tests=None,        # e.g., [(0,1), (0,2)]
    dpi=300,
    alpha_scatter=1,
    alpha_hist=0.7,
    save_path=None,
    pair_test_fun=True,
    ref_line=0,
    ref_line_color='black',
    ref_line_lw=1,
    ref_line_ls='-',
    scatter_plot=True,
    rot_label=None
):
    if large_gap_cols is None:
        large_gap_cols = []
    if sig_test_vs_zero is None:
        sig_test_vs_zero = []
    if paired_tests is None:
        paired_tests = []
    if col_names is None:
        col_names = [f"Group {i+1}" for i in range(data.shape[1])]
    if bar_colors is None:
        bar_colors = sns.color_palette("colorblind", data.shape[1])

    n_cond = data.shape[1]

    '''calculate test value '''
    df_data = pd.DataFrame(data, columns=col_names)
    
    # Analysis step.
    means = df_data.mean().values
    vars_ = df_data.var().values  # sample variance, ddof=1
    t_zero = []
    p_zero = []
    df_zero = []
    cohen_d_zero = []
    
    for idx in range(n_cond):
        # Plot or analysis configuration.
        col_data = df_data.iloc[:, idx].dropna().values
        
        col_name = col_names[idx]   # Plot or analysis configuration.
        
        n_valid = len(col_data)
        # for col in col_names:
        #     # # print('col',col)
        #     col_data = df_data[col].dropna().values
        if len(col_data) < 2:
            t_zero.append(np.nan)
            p_zero.append(np.nan)
            df_zero.append(np.nan)
            cohen_d_zero.append(np.nan)
            continue
        res = stats.ttest_1samp(col_data, 0)
        t_zero.append(res[0])#res.statistic
        # # print('col_data',col_data.shape)
        # # print('res[0]',res[0])
        # # print('res[1]',res[1])

        # # print('res.pvalue',res.pvalue)
        p_zero.append(res[1])#res.pvalue
        df_val = len(col_data) - 1
        df_zero.append(df_val)
        # Analysis step.
        cohen_d = res[0] / np.sqrt(len(col_data))
        cohen_d_zero.append(cohen_d)
    
    # Analysis step.
    corr_matrix = df_data.corr().values
    
    # Add the configured statistical comparison.
    p_paired_matrix = np.full((n_cond, n_cond), np.nan)
    t_paired_matrix = np.full((n_cond, n_cond), np.nan)
    df_paired_matrix = np.full((n_cond, n_cond), np.nan)
    cohen_d_paired_matrix = np.full((n_cond, n_cond), np.nan)
    for i in range(n_cond):
        for j in range(i + 1, n_cond):
            col1 = df_data.iloc[:, i][~np.isnan(df_data.iloc[:, i]) & ~np.isnan(df_data.iloc[:, j])]
            col2 = df_data.iloc[:, j][~np.isnan(df_data.iloc[:, i]) & ~np.isnan(df_data.iloc[:, j])]
            # # print('new new')#[~np.isnan(col1) & ~np.isnan(col2)]
            # col1 = (df_data.iloc[:, i].dropna().values)
            # col2 = df_data.iloc[:, j].dropna().values
          
            # paired_df = df_data[[col_names[i], col_names[j]]].dropna()
            # col1 = paired_df.iloc[:, 0].values
            # col2 = paired_df.iloc[:, 1].values
            
            if len(col1) < 2 or len(col2) < 2:
                continue
            # # print('col1.shape',col1[~np.isnan(col1) & ~np.isnan(col2)].shape)
            # # print('col2.shape',col2[~np.isnan(col1) & ~np.isnan(col2)].shape)
            # if len(col1) < 2:
            #     continue
                
            # try:
            # # print('col1.shape',col1.shape)
            # # print('col2.shape',col2.shape)
            #[~np.isnan(col1) & ~np.isnan(col2)]
            res = stats.ttest_rel(col1, col2)
            t = res.statistic
            p = res.pvalue
            df_val = len(col1) - 1
            cohen_d = t / np.sqrt(len(col1))
            # except:
            #     # print('nan')
            #     t = np.nan
            #     p = np.nan
            #     df_val = np.nan
            #     cohen_d = np.nan
            
            p_paired_matrix[i, j] = p
            p_paired_matrix[j, i] = p
            t_paired_matrix[i, j] = t
            t_paired_matrix[j, i] = -t if not np.isnan(t) else np.nan  # Analysis step.
            df_paired_matrix[i, j] = df_val
            df_paired_matrix[j, i] = df_val
            cohen_d_paired_matrix[i, j] = cohen_d
            cohen_d_paired_matrix[j, i] = -cohen_d if not np.isnan(cohen_d) else np.nan
    
    # Add the configured statistical comparison.
    p_ind_matrix = np.full((n_cond, n_cond), np.nan)
    t_ind_matrix = np.full((n_cond, n_cond), np.nan)
    df_ind_matrix = np.full((n_cond, n_cond), np.nan)
    cohen_d_ind_matrix = np.full((n_cond, n_cond), np.nan)
    for i in range(n_cond):
        for j in range(i + 1, n_cond):
            col1 = df_data.iloc[:, i].dropna().values
            col2 = df_data.iloc[:, j].dropna().values
            if len(col1) < 2 or len(col2) < 2:
                continue
            mean1, mean2 = np.mean(col1), np.mean(col2)
            try:
                res = stats.ttest_ind(col1, col2)
                t = res.statistic
                p = res.pvalue
                df_val = len(col1) + len(col2) - 2
                std1, std2 = np.std(col1, ddof=1), np.std(col2, ddof=1)
                pooled_std = np.sqrt(((len(col1)-1)*std1**2 + (len(col2)-1)*std2**2) / df_val) if df_val > 0 else np.nan
                cohen_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else np.nan
            except:
                t = np.nan
                p = np.nan
                df_val = np.nan
                cohen_d = np.nan
            p_ind_matrix[i, j] = p
            p_ind_matrix[j, i] = p
            t_ind_matrix[i, j] = t
            t_ind_matrix[j, i] = -t if not np.isnan(t) else np.nan  # Analysis step.
            df_ind_matrix[i, j] = df_val
            df_ind_matrix[j, i] = df_val
            cohen_d_ind_matrix[i, j] = cohen_d
            cohen_d_ind_matrix[j, i] = -cohen_d if not np.isnan(cohen_d) else np.nan

    # Calculate the requested metrics.
    base_height = 0.8
    bar_height = base_height / (n_cond + len(large_gap_cols))
    small_gap = bar_height * 0.5
    large_gap = bar_height * 2.0
    bottom_margin = bar_height / 3

    positions = []
    current_pos = bottom_margin + bar_height / 2
    positions.append(current_pos)
    for i in range(1, n_cond):
        gap = large_gap if (i-1, i) in large_gap_cols or (i, i-1) in large_gap_cols else small_gap
        current_pos += bar_height + gap
        positions.append(current_pos)
    positions = np.array(positions)

    # Initialize the result container.
    cm_to_inch = 1 / 2.54
    fig = plt.figure(figsize=(figsize_cm[0] * cm_to_inch, figsize_cm[1] * cm_to_inch))
    ax = fig.add_subplot(111)

    # Calculate the requested metrics.
    means = []
    errors = []
    scatter_y_coords = []   # Analysis step.
    max_jitter = bar_height * 0.4

    for i in range(n_cond):
        col_data = data[:, i]
        valid = col_data[~np.isnan(col_data)]
        mean = np.mean(valid)
        means.append(mean)

        if errorbar == 'sem':
            err = stats.sem(valid)
        elif errorbar == 'ci':
            ci = stats.t.interval(0.95, len(valid)-1, loc=mean, scale=stats.sem(valid))
            err = (ci[1] - ci[0]) / 2
        else:
            err = stats.sem(valid)
        errors.append(err)

        # Analysis step.
        if len(valid) > 0:
            dist = np.abs(valid - np.median(valid))
            scale = 1 - (dist / (dist.max() + 1e-8))**0.8
            jitter = np.random.uniform(-1, 1, len(valid)) * scale * max_jitter
            y_jittered = np.clip(positions[i] + jitter, positions[i] - bar_height/2, positions[i] + bar_height/2)
        else:
            y_jittered = np.array([])
        scatter_y_coords.append(y_jittered)

        if scatter_plot:
            ax.scatter(valid, y_jittered, c=[bar_colors[i]], s=13, edgecolors='none',
                       alpha=alpha_scatter, zorder=8)

    # Create the analysis plot.
    bars = ax.barh(positions, means,
                   xerr=errors,
                   capsize=4,
                   error_kw={'lw': 1, 'zorder': 20},
                   height=bar_height * 1.2,
                   color=bar_colors,
                   edgecolor='black',
                   linewidth=0.7,
                   alpha=alpha_hist,
                   zorder=3)

    # Analysis step.
    if line_plot:
        for col1, col2 in line_plot:
            y1_all = scatter_y_coords[col1]
            y2_all = scatter_y_coords[col2]
            x1 = data[:, col1][~np.isnan(data[:, col1])]
            x2 = data[:, col2][~np.isnan(data[:, col2])]
            # print('len(x1)',len(x1))
            # print('x1',x1.shape)
            # print('x2',x2.shape)
            # print('y1_all',y1_all.shape)
            # print('y2_all',y2_all.shape)
            for k in range(len(x1)):
                ax.plot([x1[k], x2[k]], [y1_all[k], y2_all[k]],
                        color='gray', alpha=0.4, lw=0.8, zorder=5)

    # Analysis step.
    def get_sig_symbol(p):
        if p >= 0.05:
            return f'p={p:.3f}'#"ns"
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        else:
            return "*"
    # def get_sig_symbol(p):
    #     if p >= 0.05:
    #         return "ns"
    #     elif p < 0.001:
    #         return "***"
    #     elif p < 0.01:
    #         return "**"
    #     else:
    #         return "*"

    max_x = np.nanmax(data)
    min_x = np.nanmin(data)
    sig_x_start = max_x + (max_x - min_x) * 0.05
    sig_x_step = (max_x - min_x) * 0.08
    current_sig_x = sig_x_start

    # Add the configured statistical comparison.
    for col_idx in sig_test_vs_zero:
        valid = data[:, col_idx][~np.isnan(data[:, col_idx])]
        p = stats.ttest_1samp(valid, 0).pvalue
        symbol = get_sig_symbol(p)
        # if symbol != "ns":
        ax.text(current_sig_x, positions[col_idx], symbol,
                ha='left', va='center', fontsize=10, fontweight='bold',rotation=-90)
        current_sig_x += sig_x_step

    # Add the configured statistical comparison.
    paired_x = sig_x_start + sig_x_step * 1.5
    for col1, col2 in paired_tests:
        
        valid1 = data[:, col1][~np.isnan(data[:, col1]) & ~np.isnan(data[:, col2])]
        valid2 = data[:, col2][~np.isnan(data[:, col1]) & ~np.isnan(data[:, col2])]
        # valid1 = data[:, col1][~np.isnan(data[:, col1] & data[:, col2])]
        # valid2 = data[:, col2][~np.isnan(data[:, col1] & data[:, col2])]
        if len(valid1) < 2:
            continue
        if pair_test_fun:
            # p = stats.ttest_rel(valid1, valid2).pvalue
            result = stats.ttest_rel(valid1, valid2)
            n = len(valid1)                    # Plot or analysis configuration.
            df = n - 1

            t_value = result.statistic
            p = result.pvalue
            p_str = f"{p:.4e}" if p < 0.0001 else f"{p:.7f}"
            # # print('2-tailed paired t-test')
            # print(f'{col_names[col1]} vs {col_names[col2]}: t({df}) = {t_value:.5f}, p = {p_str}')
        else:
            result = stats.ttest_ind(valid1, valid2)
            n = len(valid1)                    # Plot or analysis configuration.
            df = n - 1

            t_value = result.statistic
            p = result.pvalue
            p_str = f"{p:.4e}" if p < 0.0001 else f"{p:.7f}"
            # # print('2-tailed paired t-test')
            # print(f'{col_names[col1]} vs {col_names[col2]}: t({df}) = {t_value:.5f}, p = {p_str}')
            # p = stats.ttest_ind(valid1, valid2).pvalue

            # p=stats.ttest_ind(data[:, col1][~np.isnan(data[:, col1])], data[:, col2][~np.isnan(data[:, col2])])[1]
        
        # Analysis step.
        symbol = get_sig_symbol(p)
        # if symbol != "ns":
        y = positions[col1] + (positions[col2] - positions[col1]) / 2
        ax.plot([paired_x, paired_x], [positions[col1], positions[col2]],
                color='black', lw=1)
        ax.text(paired_x + sig_x_step*0.3, y, symbol,
                ha='left', va='center', fontsize=9, fontweight='bold',rotation=-90)
        paired_x += sig_x_step * 1.2

    # Configure plot appearance.
    
    ax.set_yticks(positions)
    # # print(positions)
    if rot_label is not None:
        ax.set_yticklabels(col_names,rotation=45, ha='right',va='top',rotation_mode='anchor',fontsize=11)

    else:
        ax.set_yticklabels(col_names, fontsize=11)
    
    # ax.set_yticklabels(col_names)
    ax.set_xlabel(ylabel)
    ax.invert_yaxis()  # Analysis step.
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    if ylim is not None:
        ax.set_xlim(ylim)
    if ref_line is not None:
        ax.axvline(x=ref_line, color=ref_line_color, linestyle=ref_line_ls,
                   linewidth=ref_line_lw, alpha=0.8)

    sns.despine()
    # ax.spines['bottom'].set_linewidth(1)
    # ax.spines['left'].set_linewidth(1.2)
    ax.tick_params(axis='y', which='both', length=0)
    # Plot or analysis configuration.
    # Plot or analysis configuration.
    # ax.set_xticklabels(abs_ticks_labels)
    
    ax.spines['top'].set_color('none')   # Plot or analysis configuration.
    ax.spines['right'].set_color('none')  # Plot or analysis configuration.
    ax.spines['left'].set_color('none')  # Plot or analysis configuration.
    ax.spines['bottom'].set_linewidth(0.7)  # Plot or analysis configuration.
    # Plot or analysis configuration.

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=dpi, transparent=True)
    plt.show()


# Example usage.
# np.random.seed(42)
# n_sub, n_cond = 40, 3
# data_example = np.random.randn(n_sub, n_cond) * 0.15
# Analysis step.
# Analysis step.
# data_example[:, 2] += -0.12
# # data_example[:, 3] += 0.28
# # data_example[:, 4] += 0.05
# # data_example[:, 5] += -0.20
# plot_bar_horizontal(
#     data=data_example,
#     col_names=["Baseline", "Treatment 1", "Treatment 2"],
#     ylabel="Score",
#     title="Horizontal Bar Plot with Paired Lines and Significance",
#     large_gap_cols=[(0,1)],
# Analysis step.
# Add the configured statistical comparison.
# Analysis step.
#     errorbar='sem',
#     ylim=(-1, 1),
#     ref_line=0,
#     figsize_cm=(15, 8)
# )

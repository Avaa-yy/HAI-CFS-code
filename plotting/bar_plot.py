#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon  # Analysis step.
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import pandas as pd
# Plot or analysis configuration.
# plt.rcParams['font.family'] = 'arial'
plt.rcParams['svg.fonttype']='none'
# Plot or analysis configuration.
# Plot or analysis configuration.
# plt.rcParams['svg.text.as_path'] = False
# plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def plot_bar(
    data, 
    figsize_cm=(12, 8),
    line_plot=None,
    ylim=None,
    large_gap_cols=None,
    col_names=None,
    ylabel="Value",
    title=None,
    bar_colors=None,
    errorbar='sem',  # Analysis step.
    sig_test_vs_zero=None,
    paired_tests=None,
    dpi=300,
    alpha_scatter=1,
    alpha_hist=0.5,
    save_path=None,
    pair_test_fun=True,
    ref_line=0,          # Plot or analysis configuration.
    ref_line_color='black',   # Plot or analysis configuration.
    ref_line_lw=1,        # Plot or analysis configuration.
    ref_line_ls='-',       # Plot or analysis configuration.
    pv_caled=None,
    scatter_plot=True,
    rot_label=None
):
    """Perform the documented analysis step."""
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    if large_gap_cols is None:
        large_gap_cols = []
    if sig_test_vs_zero is None:
        sig_test_vs_zero = []
    if paired_tests is None:
        paired_tests = []
    if col_names is None:
        col_names = [f"Col {i+1}" for i in range(data.shape[1])]
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
            # paired_df = df_data[[col_names[i], col_names[j]]].dropna()
            # col1 = paired_df.iloc[:, 0].values
            # col2 = paired_df.iloc[:, 1].values
            
            if len(col1) < 2:
                continue
                
            try:
                res = stats.ttest_rel(col1, col2)
                t = res.statistic
                p = res.pvalue
                df_val = len(col1) - 1
                cohen_d = t / np.sqrt(len(col1))
            except:
                t = np.nan
                p = np.nan
                df_val = np.nan
                cohen_d = np.nan
            
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
    base_width = 0.8
    violin_width = base_width / (n_cond + len(large_gap_cols))
    small_gap = violin_width * 0.5
    large_gap = violin_width * 1.0
    left_margin = violin_width / 3

    positions = []
    current_pos = left_margin + violin_width / 2
    positions.append(current_pos)
    for i in range(1, n_cond):
        gap = large_gap if (i-1, i) in large_gap_cols or (i, i-1) in large_gap_cols else small_gap
        current_pos += violin_width + gap
        positions.append(current_pos)
    positions = np.array(positions)

    # Initialize the result container.
    fig = plt.figure(figsize=(1, 1))
    cm_to_inch = 1 / 2.54
    fig.set_size_inches(figsize_cm[0] * cm_to_inch, figsize_cm[1] * cm_to_inch)
    ax = fig.add_subplot(111)

    # Calculate the requested metrics.
    means = []
    errors = []
    scatter_x_coords = []
    max_jitter = violin_width * 0.4

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
            err = np.std(valid) / np.sqrt(len(valid))  # Analysis step.
        
        if pv_caled:
            # print('pv_caled',pv_caled)
            # print('pv_caled[1]',pv_caled[1])
            errors.append(pv_caled[1][i])
        else:
            errors.append(err)
        # # print('errors',errors)
        # Analysis step.
        if len(valid) > 0:
            dist = np.abs(valid - np.median(valid))
            scale = 1 - (dist / (dist.max() + 1e-8))**0.8
            jitter = np.random.uniform(-1, 1, len(valid)) * scale * max_jitter
            x_jittered = np.clip(positions[i] + jitter, positions[i] - violin_width/2, positions[i] + violin_width/2)
        else:
            x_jittered = np.array([])
        scatter_x_coords.append(x_jittered)
        if scatter_plot:
            ax.scatter(x_jittered, valid, c=[bar_colors[i]], s=16, edgecolors='none', alpha=alpha_scatter, zorder=8)

    # Create the analysis plot.
    bars = ax.bar(positions, means, 
              yerr=errors,           # Plot or analysis configuration.
              capsize=4, #capthick=2, 
              error_kw={'lw': 1, 'zorder': 20},  # Plot or analysis configuration.
              width=violin_width*1.2,
              color=bar_colors, 
              edgecolor='black', 
              linewidth=1, 
              alpha=alpha_hist, 
              zorder=3)   # Plot or analysis configuration.
    # ax.bar(positions, means, width=violin_width*1.2,
    # Plot or analysis configuration.
    # bars = ax.bar(positions, means, yerr=errors, width=violin_width*1.2,
    #               capsize=5, color=bar_colors, edgecolor='black', linewidth=1, alpha=alpha_hist, zorder=20)
    
    
    # Plot or analysis configuration.
    # for bar in bars:
    # Plot or analysis configuration.
    # Plot or analysis configuration.
        # Plot or analysis configuration.
        # Plot or analysis configuration.
        # Plot or analysis configuration.
        # Plot or analysis configuration.
    # Analysis step.
    if line_plot:
        # Plot or analysis configuration.
        for col1, col2 in line_plot:
            x1_all = scatter_x_coords[col1]   # Plot or analysis configuration.
            x2_all = scatter_x_coords[col2]
            y1 = data[:, col1][~np.isnan(data[:, col1])]#data[:, col1]
            y2 = data[:, col2][~np.isnan(data[:,col2])]#data[:, col2]
            for k in range(len(y1)):
                ax.plot([x1_all[k], x2_all[k]], [y1[k], y2[k]], 
                        color='gray', alpha=0.25, lw=0.8, zorder=5)
    # Add the configured statistical comparison.
    def get_sig_symbol(p):
        if p >= 0.05:
            return f'p={p:.3f}'#"ns"
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        else:
            return "*"
        
        
    max_y = np.nanmax(data)#data.max() #if ylim is None else ylim[1]
    min_y=np.nanmin(data)

    sig_y_start = max_y + (max_y - min_y) * 0.02 #if ylim is None else ylim[1] * 1.05
    sig_y_step = (max_y - min_y) * 0.05 #if ylim is None else (ylim[1] - ylim[0]) * 0.08
    # # print('max_y ',max_y )
    # # print('max_y - min_y',max_y - min_y)
    # # print('sig_y_start',sig_y_start)
    
    current_sig_y = sig_y_start


    if pv_caled:
        # Analysis step.
        sig_zero_ys = {}
        for col_idx in sig_test_vs_zero:
            p = stats.ttest_1samp(data[:, col_idx][~np.isnan(data[:, col_idx])], 0)[1]  # Analysis step.
            symbol = get_sig_symbol(pv_caled[0][col_idx])
            # if symbol != "ns":
            # # print('positions[col_idx], current_sig_y',positions[col_idx], current_sig_y)
            ax.text(positions[col_idx], current_sig_y, symbol, ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
            sig_zero_ys[col_idx] = current_sig_y
    else:
        # Analysis step.
        sig_zero_ys = {}
        for col_idx in sig_test_vs_zero:
            p = stats.ttest_1samp(data[:, col_idx][~np.isnan(data[:, col_idx])], 0)[1]  # Analysis step.
            symbol = get_sig_symbol(p)
            # if symbol != "ns":
            ax.text(positions[col_idx], current_sig_y, symbol, ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
            sig_zero_ys[col_idx] = current_sig_y
            # current_sig_y += sig_y_step

    # Add the configured statistical comparison.
    paired_y = current_sig_y + sig_y_step# * 0.5

    for col1, col2 in paired_tests:
        if pair_test_fun:
            valid=(~np.isnan(data[:, col2]))&(~np.isnan(data[:, col1]))
            
            xx=data[:, col1][valid]
            yy=data[:, col2][valid]
            result = stats.ttest_rel(xx, yy)
            n = len(xx)                    # Plot or analysis configuration.
            df = n - 1

            t_value = result.statistic
            p = result.pvalue
            p_str = f"{p:.4e}" if p < 0.0001 else f"{p:.7f}"
            # # print('2-tailed paired t-test')
            # print(f'{col_names[col1]} vs {col_names[col2]}: t({df}) = {t_value:.3f}, p = {p_str}')
            # p = stats.ttest_rel(data[:, col1][valid], data[:, col2][valid])[1]

            # # print('2-tailed paired t-test',)
        else:
            valid=(~np.isnan(data[:, col2]))&(~np.isnan(data[:, col1]))
            
            xx=data[:, col1][valid]
            yy=data[:, col2][valid]
            result = stats.ttest_ind(xx, yy)
            n = len(xx)                    # Plot or analysis configuration.
            df = n - 1

            t_value = result.statistic
            p = result.pvalue
            p_str = f"{p:.4e}" if p < 0.0001 else f"{p:.7f}"
            # # print('2-tailed paired t-test')
            # print(f'{col_names[col1]} vs {col_names[col2]}: t({df}) = {t_value:.3f}, p = {p_str}')
            # # print('2-tailed t-test')
            # # print(f'{col_names[col1]} vs {col_names[col2]}: t({df}) = {t_value}, p = {p}')
            
            
            # p=stats.ttest_ind(data[:, col1][valid], data[:, col2][valid])[1]
        # Analysis step.
        symbol = get_sig_symbol(p)
        # if symbol != "ns":
        x1, x2 = positions[col1], positions[col2]
        y = paired_y
        ax.plot([x1, x2], [y+sig_y_step, y+sig_y_step], color='black', lw=1)
        # ax.plot([x1, x1, x2, x2], [y, y+sig_y_step, y+sig_y_step, y], 
        #         lw=1, c='black')
        ax.text((x1+x2)/2, y+sig_y_step, symbol, ha='center', va='bottom', 
                fontsize=8, fontweight='bold')
        paired_y += sig_y_step * 0.8

    # Configure plot appearance.
    ax.set_xticks(positions)
    if rot_label is not None:
        ax.set_xticklabels(col_names,rotation=45, ha='right',va='top',rotation_mode='anchor',fontsize=11)

    else:
        ax.set_xticklabels(col_names, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    # Plot or analysis configuration.
    if ref_line is not None:
        ax.axhline(y=ref_line, color=ref_line_color, linestyle=ref_line_ls,
                   linewidth=ref_line_lw, alpha=0.8)

    
    sns.despine()
    ax.spines['bottom'].set_linewidth(0.7)
    ax.spines['left'].set_linewidth(0.7)

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=dpi, transparent=True)

    plt.show()
    
# Plot or analysis configuration.

# Plot or analysis configuration.

# np.random.seed(42)
# n_sub, n_cond = 40, 6
# data_example = np.random.randn(n_sub, n_cond) * 0.15
# Analysis step.
# Analysis step.
# data_example[:, 2] += -0.12
# data_example[:, 3] += 0.28
# data_example[:, 4] += 0.05
# data_example[:, 5] += -0.20

# Example usage.
# plot_bar(
#     data=data_example,
#     figsize_cm=(14, 9),
#     ylim=(-0.6, 1),
#     line_plot=[(2,3)],
# Analysis step.
#     col_names=['s', 'd', 's', 'd', 's', 'd'],
#     ylabel='SOM effect',
#     title='SOM Effect across ROIs',
#     bar_colors=[student_colors[0],doctor_colors[0],student_colors[1],doctor_colors[1],student_colors[2],doctor_colors[2], ],
# Analysis step.
# Add the configured statistical comparison.
#     save_path='violin_som_example.svg',
#     alpha_hist=0.6
# )

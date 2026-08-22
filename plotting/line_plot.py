#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon  # Analysis step.
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
# Plot or analysis configuration.
# plt.rcParams['font.family'] = 'arial'
# plt.rcParams['svg.fonttype']='none'
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


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def plot_line(
    data_list,                              # Analysis step.
    figsize_cm=(14, 8),
    ylim=None,
    xlim=None,
    large_gap_cols=None,
    col_names=None,
    ylabel="Value",
    xlabel=None,
    title=None,
    line_colors=None,                       # Configure plot appearance.
    errorbar='sem',                         # Analysis step.
    sig_test_vs_zero=None,                  # Add the configured statistical comparison.
    paired_tests=None,                      # Add the configured statistical comparison.
    dpi=300,
    label_list=None,
    save_path=None,
    ref_line=None,
    ref_vertical_x=None,
    ref_line_color='black',
    ref_line_lw=1.5,
    ref_line_ls='-',
    marker='o',
    markersize=8,
    linewidth=3,
    capsize=6,
    diag_plot=None,
    diag_col=None,
    rot_label=None,
    position_my=None 
):
    """Perform the documented analysis step."""
    if large_gap_cols is None:
        large_gap_cols = []
    if sig_test_vs_zero is None:
        sig_test_vs_zero = []
    if paired_tests is None:
        paired_tests = []
    if col_names is None:
        col_names = [f"X{i+1}" for i in range(len(data_list[0]))]
    if line_colors is None:
        line_colors = sns.color_palette("tab10", len(data_list))

    n_feature = data_list[0].shape[1]# Analysis step.
    n_group = len(data_list)         # Analysis step.

    # Calculate the requested metrics.
    base_width = 0.8
    violin_width = base_width / (n_feature + len(large_gap_cols))
    small_gap = violin_width * 0.5
    large_gap = violin_width * 1.0
    left_margin = violin_width# / 2
    # print('small_gap ',small_gap )
    # print('left_margin ',left_margin )
    # print('left_margin + violin_width / 2',left_margin + violin_width / 2)
    
    positions = []
    current_pos = left_margin + violin_width / 2
    positions.append(current_pos)
    for i in range(1, n_feature):
        gap = large_gap if (i-1, i) in large_gap_cols or (i, i-1) in large_gap_cols else small_gap
        current_pos += violin_width/2 + gap
        positions.append(current_pos)
    positions = np.array(positions)
    if position_my:
        positions=position_my
    # print('positions',positions)
    # Initialize the result container.
    fig = plt.figure(figsize=(1, 1))
    cm_to_inch = 1 / 2.54
    fig.set_size_inches(figsize_cm[0] * cm_to_inch, figsize_cm[1] * cm_to_inch)
    ax = fig.add_subplot(111)

    # Calculate the requested metrics.
    all_means = []
    all_errors = []
    for data in data_list:
        means = []
        errors = []
        # # print('n_feature',n_feature)
        for i in range(n_feature):
            # # print('i',i)
            col = data[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                means.append(np.nan)
                errors.append(0)
                continue
            mean = np.mean(valid)
            means.append(mean)
            if errorbar == 'sem':
                err = stats.sem(valid)
            elif errorbar == 'ci':
                ci = stats.t.interval(0.95, len(valid)-1, loc=mean, scale=stats.sem(valid))
                err = (ci[1] - ci[0]) / 2 if len(valid) > 1 else 0
            else:
                err = stats.sem(valid)
            errors.append(err)
        all_means.append(means)
        all_errors.append(errors)

    all_means = np.array(all_means)   # shape: (n_group, n_feature)
    all_errors = np.array(all_errors)

    # Analysis step.
    for g in range(n_group):
        color = line_colors[g]
        
        mean_line = all_means[g]          # shape: (n_feature,)
        err      = all_errors[g]          # shape: (n_feature,)
        # print('err',err)
        # Plot or analysis configuration.
        ax.fill_between(
            positions, 
            mean_line - err, 
            mean_line + err,
            color=color, 
            alpha=0.3,          # Plot or analysis configuration.
            linewidth=0,         # Plot or analysis configuration.
            zorder=3             # Plot or analysis configuration.
        )
        if label_list:
            ax.plot(positions, all_means[g], 
                    marker=marker, markersize=markersize, linewidth=linewidth,
                    color=color, label=label_list[g], zorder=5)
        else:
            ax.plot(positions, all_means[g], 
                    marker=marker, markersize=markersize, linewidth=linewidth,
                    color=color,  zorder=5)
        # # errorbar
        # ax.errorbar(positions, all_means[g], yerr=all_errors[g],
        #             fmt='none', ecolor=color, alpha=0.8,
        #             capsize=capsize, capthick=linewidth*0.6, linewidth=linewidth*0.7, zorder=4)

    # Add the configured statistical comparison.
    def get_sig_symbol(p):
        if p >= 0.05: return "n.s."
        elif p < 0.001: return "***"
        elif p < 0.01: return "**"
        else: return "*"

    global_max = np.nanmax(all_means + all_errors)
    global_min = np.nanmin(all_means - all_errors)
    if ylim:
        plot_max = ylim[1]
    else:
        plot_max = global_max + (global_max - global_min) * 0.1

    sig_y = plot_max + (global_max - global_min) * 0.08
    step = (global_max - global_min) * 0.1

    # Add the configured statistical comparison.
    for col_idx in sig_test_vs_zero:
        for g, data in enumerate(data_list):
            valid = data[:, col_idx][~np.isnan(data[:, col_idx])]
            if len(valid) > 1:
                p = stats.ttest_1samp(valid, 0).pvalue
                symbol = get_sig_symbol(p)
                x_pos = positions[col_idx]
                ax.text(x_pos, sig_y + g*step*0.6, symbol,
                        ha='center', va='bottom', color=line_colors[g],
                        fontsize=11, fontweight='bold')
        sig_y += step

    # Add the configured statistical comparison.
    current_y = sig_y
    for col1, col2 in paired_tests:
        for g in range(n_group):
            y1 = data_list[g][:, col1]
            y2 = data_list[g][:, col2]
            v1 = y1[~np.isnan(y1)]
            v2 = y2[~np.isnan(y2)]
            if len(v1) > 1 and len(v2) > 1 and len(v1) == len(v2):
                p = stats.ttest_rel(v1, v2).pvalue
                symbol = get_sig_symbol(p)
                x_mid = (positions[col1] + positions[col2]) / 2
                ax.text(x_mid, current_y, symbol,
                        ha='center', va='bottom', color=line_colors[g],
                        fontsize=10, fontweight='bold')
        current_y += step * 0.8

    # Add the configured statistical comparison.

    # Configure plot appearance.
    ax.set_xticks(positions)
    if rot_label is not None:
        ax.set_xticklabels(col_names,rotation=45, ha='right',va='top',rotation_mode='anchor')

    else:
        ax.set_xticklabels(col_names)
    # ax.set_xticklabels(col_names)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, pad=20)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    # else:
    #     ax.set_xlim((positions[0],))
    if ref_line is not None:
        if ref_vertical_x is not None:
            ax.axvline(x=ref_line, color=ref_line_color, linestyle=ref_line_ls,
                       linewidth=ref_line_lw, alpha=0.9, zorder=1)
        else:
            ax.axhline(y=ref_line, color=ref_line_color, linestyle=ref_line_ls,
                       linewidth=ref_line_lw, alpha=0.9, zorder=1)
    if diag_plot:
        ax.plot([positions[0], positions[-1]], [0, 1], color=diag_col, linestyle='-',linewidth=linewidth, alpha=0.9, zorder=1)
    sns.despine()
    ax.spines['bottom'].set_linewidth(0.7)
    ax.spines['left'].set_linewidth(0.7)

    # Analysis step.
    ax.legend(frameon=False,  loc='upper left')

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=dpi, transparent=True)
        # print(f"Multi-group line plot saved: {save_path}")

    plt.show()
# Apply the configured random split.
# np.random.seed(42)
# n_feature = 5
# col_names = ['0', '5', '10', '15', '20']

# Analysis step.
# data1 = np.random.normal(loc=50, scale=15, size=(30, n_feature))
# Analysis step.

# Analysis step.
# data2 = np.random.normal(loc=48, scale=18, size=(25, n_feature))
# Analysis step.

# Analysis step.
# data3 = np.random.normal(loc=52, scale=12, size=(20, n_feature))
# Analysis step.

# Analysis step.
# data_list = [data1, data2, data3]

# Create the analysis plot.
# plot_line(
#     data_list=data_list,
#     figsize_cm=(9, 7),
#     col_names=col_names,
#     label_list=['NoConf','Softmax','Meta'],
#     ylabel='change ratio',
#     title='',
# Analysis step.
#     errorbar='sem',
#     ylim=(20, 100),
# Analysis step.
# Analysis step.
# Analysis step.
#     linewidth=2,
#     markersize=3,
#     save_path='multi_group_line_demo.svg'
# )

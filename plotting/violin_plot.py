
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon  # Analysis step.
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
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

def plot_violin(
    data,                              # (n_subjects, n_conditions) array
    figsize_cm=(12, 8),               # Create the analysis plot.
    line_plot=None,
    ylim=None,                         # Analysis step.
    large_gap_cols=None,               # Analysis step.
    col_names=None,                    # Analysis step.
    ylabel="SOM effect",               # Analysis step.
    title=None,                        # Analysis step.
    violin_colors=None,                # Configure plot appearance.
    sig_test_vs_zero=None,             # Analysis step.
    paired_tests=None,                 # Add the configured statistical comparison.
    dpi=300,
    alpha_v=0.4,
    ref_line=0,          # Plot or analysis configuration.
    ref_line_color='black',   # Plot or analysis configuration.
    ref_line_lw=1,        # Plot or analysis configuration.
    ref_line_ls='-',       # Plot or analysis configuration.
    save_path=None,                    # Save the generated result.
    pair_test_fun=True
):
    """
    Advanced violin plot with boxplot, individual points, significance bars, and customizable spacing.
    """
    if large_gap_cols is None:
        large_gap_cols = []
    if sig_test_vs_zero is None:
        sig_test_vs_zero = []
    if paired_tests is None:
        paired_tests = []
    if col_names is None:
        col_names = [f"Col {i+1}" for i in range(data.shape[1])]
    if violin_colors is None:
        violin_colors = sns.color_palette("colorblind", data.shape[1])

    n_cond = data.shape[1]
    
    # alpha_v=0.4
    # Calculate the requested metrics.
    base_width = 0.8
    violin_width = base_width / (n_cond + (len(large_gap_cols) * 1))  # Analysis step.
    small_gap = violin_width * (1/2)      # Analysis step.
    large_gap = violin_width * 1        # Analysis step.
    left_margin = violin_width / 3        # Analysis step.

    # Calculate the requested metrics.
    positions = []
    current_pos = left_margin + violin_width / 2
    positions.append(current_pos)
    
    for i in range(1, n_cond):
        gap = large_gap if (i-1, i) in large_gap_cols or (i, i-1) in large_gap_cols else small_gap
        current_pos += violin_width + gap
        positions.append(current_pos)
    positions = np.array(positions)

    # Create the analysis plot.
    fig = plt.figure(figsize=(1, 1))  # Analysis step.
    cm_to_inch = 1 / 2.54
    fig.set_size_inches(figsize_cm[0] * cm_to_inch, figsize_cm[1] * cm_to_inch)
    ax = fig.add_subplot(111)
    # Plot or analysis configuration.
    # Analysis step.
    parts = ax.violinplot(
        [data[:, i][~np.isnan(data[:, i])] for i in range(n_cond)],
        positions=positions,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for pc, color in zip(parts['bodies'], violin_colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
        pc.set_alpha(alpha_v)

    # Plot or analysis configuration.
    
        # Plot or analysis configuration.
    scatter_x_coords = []  # Plot or analysis configuration.
    max_jitter = violin_width * 0.4

    # Plot or analysis configuration.
    for i in range(n_cond):
        x0 = positions[i]
        y = data[:, i][~np.isnan(data[:, i])]#data[:, i]
        dist = np.abs(y - np.median(y))
        scale = 1 - (dist / (dist.max() + 1e-8))**0.8
        jitter = np.random.uniform(-1, 1, len(y)) * scale * max_jitter
        x_jittered = np.clip(x0 + jitter, x0 - violin_width/2, x0 + violin_width/2)
        scatter_x_coords.append(x_jittered)   # Plot or analysis configuration.

        # Plot or analysis configuration.
        ax.scatter(x_jittered, y, c=violin_colors[i], s=14, 
                   edgecolors='none', alpha=0.7, zorder=10)
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
    # Apply the configured random split.
    # Plot or analysis configuration.
    # max_jitter = violin_width * 0.4
    
    # for i in range(n_cond):
    #     x0 = positions[i]
    #     y = data[:, i]
    #     dist = np.abs(y - np.median(y))
    # Plot or analysis configuration.
    #     jitter = np.random.uniform(-1, 1, len(y)) * scale * max_jitter
    #     x = np.clip(x0 + jitter, x0 - violin_width/2, x0 + violin_width/2)
    #     ax.scatter(x, y, c=violin_colors[i], s=14, edgecolors='none',alpha=0.7)
    
    # Plot or analysis configuration.
    # Plot or analysis configuration.
    #     for col1, col2 in line_plot:
    #         x1 = positions[col1]
    #         x2 = positions[col2]
    #         y1 = data[:, col1]
    #         y2 = data[:, col2]
    # Plot or analysis configuration.
    #         for i in range(len(y1)):
    #             ax.plot([x1, x2], [y1[i], y2[i]], 
    #                     color='grey', alpha=alpha_v, lw=0.8, zorder=5)
    
    
    # jitter = violin_width * 0.2
    # for i in range(n_cond):
    #     x = positions[i] + np.random.uniform(-jitter, jitter, size=data.shape[0])
    #     ax.scatter(x, data[:, i], c=[violin_colors[i]], s=12, edgecolors='none', alpha=1)

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

    # Create the analysis plot.
    for i in range(n_cond):
        q1, med, q3 = np.percentile(data[:, i][~np.isnan(data[:, i])], [25, 50, 75])
        iqr = q3 - q1
        whisker_low  = np.max([q1 - 1.5*iqr, data[:, i][~np.isnan(data[:, i])].min()])
        whisker_high = np.min([q3 + 1.5*iqr, data[:, i][~np.isnan(data[:, i])].max()])

        # Analysis step.
        ax.plot([positions[i]-violin_width/6, positions[i]+violin_width/6], [med, med], 
                color='black', lw=1,alpha=alpha_v,zorder=20)
        ax.plot([positions[i]-violin_width/6, positions[i]-violin_width/6], [q1, q3], 
                color='black', lw=1,alpha=alpha_v,zorder=20)
        ax.plot([positions[i]+violin_width/6, positions[i]+violin_width/6], [q1, q3], 
                color='black', lw=1,alpha=alpha_v,zorder=20)
        ax.plot([positions[i]-violin_width/6, positions[i]+violin_width/6], [q1, q1], 
                color='black', lw=1,alpha=alpha_v,zorder=20)
        ax.plot([positions[i]-violin_width/6, positions[i]+violin_width/6], [q3, q3], 
                color='black', lw=1,alpha=alpha_v,zorder=20)

        # Analysis step.
        # ax.plot([positions[i]-violin_width/8, positions[i]+violin_width/8], [whisker_low, whisker_low], 
        #         color='black', lw=1,alpha=alpha_v)
        # ax.plot([positions[i]-violin_width/8, positions[i]+violin_width/8], [whisker_high, whisker_high], 
        #         color='black', lw=1,alpha=alpha_v)
        ax.plot([positions[i], positions[i]], [whisker_low, q1], color='black', lw=1,alpha=alpha_v,zorder=20)
        ax.plot([positions[i], positions[i]], [q3, whisker_high], color='black', lw=1,alpha=alpha_v,zorder=20)


    max_y = np.nanmax(data)#data.max() #if ylim is None else ylim[1]
    min_y=np.nanmin(data)
    sig_y_start = max_y + (max_y - min_y) * 0.02 #if ylim is None else ylim[1] * 1.05
    sig_y_step = (max_y - min_y) * 0.05 #if ylim is None else (ylim[1] - ylim[0]) * 0.08

    current_sig_y = sig_y_start

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
            p = stats.ttest_rel(data[:, col1][~np.isnan(data[:, col1])], data[:, col2][~np.isnan(data[:, col2])])[1]
        else:
            p=stats.ttest_ind(data[:, col1][~np.isnan(data[:, col1])], data[:, col2][~np.isnan(data[:, col2])])[1]
        # Analysis step.
        symbol = get_sig_symbol(p)
        # if symbol != "ns":
        x1, x2 = positions[col1], positions[col2]
        y = paired_y
        ax.plot([x1, x2], [y+sig_y_step, y+sig_y_step], color='black', lw=1)
        # ax.plot([x1, x1, x2, x2], [y, y+sig_y_step, y+sig_y_step, y], 
        #         lw=1, c='black')
        ax.text((x1+x2)/2, y+sig_y_step, symbol, ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
        paired_y += sig_y_step * 0.8

    # Analysis step.
    ax.set_xticks(positions)
    ax.set_xticklabels(col_names, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, pad=15)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Analysis step.
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # ax.tick_params(axis='both', which='major', labelsize=10)
    if ref_line is not None:
        ax.axhline(y=ref_line, color=ref_line_color, linestyle=ref_line_ls,
                   linewidth=ref_line_lw, alpha=0.8)
    # Analysis step.
    sns.despine()

    # Save the generated result.
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=dpi, transparent=True)
        # print(f"Plot saved to {save_path}")

    plt.show()


# # ==============================
# Create the analysis plot.
# # ==============================
# np.random.seed(42)
# n_sub, n_cond = 40, 4
# data_example = np.random.randn(n_sub, n_cond) * 0.15
# Analysis step.
# Analysis step.
# data_example[:, 2] += -0.12
# data_example[:, 3] += 0.28
# data_example[:, 4] += 0.05
# data_example[:, 5] += -0.20

# Example usage.
# plot_violin(
#     data=data_example,
#     figsize_cm=(14, 9),
#     line_plot=[(1, 2)],
#     ylim=(-0.6, 1),
# Analysis step.
#     col_names=['Vertex', 'rTPJ', 'LTPJ', 'mPFC'],
#     ylabel='SOM effect',
#     title='SOM Effect across ROIs',
#     violin_colors=['#0072B2', '#D55E00', '#CC79A7', '#009E73'],
# Analysis step.
# Add the configured statistical comparison.
#     save_path='violin_som_example.svg'
# )
# # In[]
# Plot or analysis configuration.
# Plot or analysis configuration.

# Plot or analysis configuration.
# Plot or analysis configuration.

# Plot or analysis configuration.
# Plot or analysis configuration.

# Plot or analysis configuration.
# Plot or analysis configuration.

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
# plot_violin(
#     data=data_example,
#     figsize_cm=(14, 9),
#     ylim=(-0.6, 1),
#     line_plot=[(2,3)],
# Analysis step.
#     col_names=['s', 'd', 's', 'd', 's', 'd'],
#     ylabel='SOM effect',
#     title='SOM Effect across ROIs',
#     violin_colors=[student_colors[0],doctor_colors[0],student_colors[1],doctor_colors[1],student_colors[2],doctor_colors[2], ],
# Analysis step.
# Add the configured statistical comparison.
#     save_path='violin_som_example.svg',
#     alpha_v=0.6
# )
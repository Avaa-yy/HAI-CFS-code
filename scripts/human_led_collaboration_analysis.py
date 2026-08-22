#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plotting.violin_plot import plot_violin
from plotting.bar_plot import plot_bar
import pandas as pd
from statsmodels.formula.api import logit

DATA_DIR = PROJECT_ROOT / 'data'
FILES_DIR = PROJECT_ROOT / 'files'
FIGURES_DIR = PROJECT_ROOT / 'figures'
FILES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
def remove_outliers_iqr(df, column, factor=1.5):
    """Remove values outside the configured interquartile range."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]
d_col = ['#2E54A1', '#8B6F47', '#BA3E45']
s_col = ['#81B1D6', '#F4A15D', '#F19CBB']
# In[1-1]
'''calculate initial and final accuracy for each sub*each group'''
import pandas as pd

# Load the required data.
file_path = DATA_DIR / 'retinal_diagnosis_trials.xlsx'
df = pd.read_excel(file_path)
df = df[df['ai_initial_agreement'] == 0]
# Initialize the result container.
results = []

# Analysis step.
for id in range(1, 50):
    for experiment in [1, 2, 3]:
        # Filter the relevant observations.
        filtered_df = df[(df["participant_id"] == id) & (df["confidence_condition"] == experiment)]
        
        # Calculate the requested metrics.
        if not filtered_df.empty:
            ai_accuracy = filtered_df["ai_correct"].mean()
            initial_accuracy = filtered_df["initial_correct"].mean()
            final_accuracy = filtered_df["final_correct"].mean()
            
            # Analysis step.
            results.append([id,  filtered_df['participant_group'].iloc[0],experiment, ai_accuracy, initial_accuracy, final_accuracy])

# Initialize the result container.
results_df = pd.DataFrame(results, columns=["participant_id", 'participant_group',"confidence_condition", "ai_correct", "initial_correct", "final_correct"])

# Save the generated result.
output_file_path = FILES_DIR / 'participant_accuracy_by_condition.xlsx'
results_df.to_excel(output_file_path, index=False)

# In[1-2]
'''plot initial and final accuracy for each sub*each group'''

import pandas as pd
import numpy as np

# identi='medical_student'
identi='ophthalmologist'

if identi=='medical_student':
    identi_name='Medical student'
    identi_slug='medical_student'
    y_range=(0, 1)
    color_list=[s_col[0],s_col[0],s_col[0],s_col[1],s_col[1],s_col[1],s_col[2],s_col[2],s_col[2]]

else:
    identi_name='Ophthalmologist'
    identi_slug='ophthalmologist'
    y_range=(0, 1.05)
    color_list=[d_col[0],d_col[0],d_col[0],d_col[1],d_col[1],d_col[1],d_col[2],d_col[2],d_col[2]]

# Load the required data.
df = pd.read_excel(FILES_DIR / 'participant_accuracy_by_condition.xlsx')
df.columns = df.columns.str.strip()  # Analysis step.

# Analysis step.
med = df[df['participant_group'] == identi].copy()

# Filter the relevant observations.
g1 = med[med['confidence_condition'] == 1].sort_values('participant_id').reset_index(drop=True)
g2 = med[med['confidence_condition'] == 2].sort_values('participant_id').reset_index(drop=True)
g3 = med[med['confidence_condition'] == 3].sort_values('participant_id').reset_index(drop=True)


# Analysis step.
max_n = max(len(g1), len(g2), len(g3))

def pad_with_nan(series, target_len):
    """Perform the documented analysis step."""
    if len(series) >= target_len:
        return series.iloc[:target_len].values
    else:
        pad = np.full(target_len - len(series), np.nan)
        return np.concatenate([series.values, pad])

# Analysis step.
data_9col = np.column_stack([
    # Analysis step.
    pad_with_nan(g1['initial_correct'], max_n),
    pad_with_nan(g1['final_correct'], max_n),
    pad_with_nan(g1['ai_correct'],   max_n),
    # Analysis step.
    pad_with_nan(g2['initial_correct'], max_n),
    pad_with_nan(g2['final_correct'], max_n),
    pad_with_nan(g2['ai_correct'],   max_n),
    # Analysis step.
    pad_with_nan(g3['initial_correct'], max_n),
    pad_with_nan(g3['final_correct'], max_n),
    pad_with_nan(g3['ai_correct'],   max_n)
])

# Save the generated result.


# Create the analysis plot.
result_accuracy = data_9col
plot_bar(
    data=result_accuracy,
    figsize_cm=(10, 6),
    ylim=y_range,
    line_plot=[],
    large_gap_cols=[(2,3),(5,6)],
    col_names=['init', 'final', 'AI', 'init', 'final', 'AI','init', 'final', 'AI'],
    ylabel='Accuracy',
    title='',
    bar_colors= color_list,
    sig_test_vs_zero=[],
    paired_tests=[(0,1),(3,4),(6,7),(1,2),(4,5),(7,8)],
    errorbar='sem',  # Plot or analysis configuration.
    alpha_scatter=1,
    alpha_hist=0.5,
    save_path=FIGURES_DIR / f'human_led_collaboration_{identi_slug}_accuracy.svg',
    scatter_plot=True
)

# In[1-4]
'''plot AI diff for each sub*each group'''

import pandas as pd
import numpy as np

# identi='medical_student'
identi='ophthalmologist'

if identi=='medical_student':
    identi_name='Medical student'
    identi_slug='medical_student'
    y_range=(-0.5, 0.3)
    color_list=s_col

else:
    identi_name='Ophthalmologist'
    identi_slug='ophthalmologist'
    y_range=(-0.6, 0.3)
    color_list=d_col


# Load the required data.
df = pd.read_excel(FILES_DIR / 'participant_accuracy_by_condition.xlsx')
df.columns = df.columns.str.strip()  # Analysis step.

# Analysis step.
med = df[df['participant_group'] == identi].copy()

# Filter the relevant observations.
g1 = med[med['confidence_condition'] == 1].sort_values('participant_id').reset_index(drop=True)
g2 = med[med['confidence_condition'] == 2].sort_values('participant_id').reset_index(drop=True)
g3 = med[med['confidence_condition'] == 3].sort_values('participant_id').reset_index(drop=True)


# Analysis step.
max_n = max(len(g1), len(g2), len(g3))

def pad_with_nan(series, target_len):
    """Perform the documented analysis step."""
    if len(series) >= target_len:
        return series.iloc[:target_len].values
    else:
        pad = np.full(target_len - len(series), np.nan)
        return np.concatenate([series.values, pad])

# Analysis step.
data_9col = np.column_stack([
    # Analysis step.
    pad_with_nan(g1['final_correct']-g1['ai_correct'], max_n),
    # pad_with_nan(, max_n),
    # pad_with_nan(g1['ai_correct'],   max_n),
    # Analysis step.
    pad_with_nan(g2['final_correct']-g2['ai_correct'], max_n),
    # pad_with_nan(, max_n),
    # pad_with_nan(g2['ai_correct'],   max_n),
    # Analysis step.
    pad_with_nan(g3['final_correct']-g3['ai_correct'], max_n),
    # pad_with_nan(, max_n),
    # pad_with_nan(g3['ai_correct'],   max_n)
])

# Save the generated result.

# Create the analysis plot.
result_accuracy = data_9col


plot_bar(
    data=result_accuracy,
    figsize_cm=(7, 6),
    ylim=y_range,
    line_plot=[(0,1),(1,2)],
    large_gap_cols=[],
    col_names=['NoConf','Softmax','Meta'],
    ylabel=f'{identi_name} acc improvement',
    title='',
    bar_colors=color_list,
    sig_test_vs_zero=[0,1,2],
    paired_tests=[(0,1),(1,2),(0,2)],
    errorbar='sem',  # Plot or analysis configuration.
    alpha_scatter=1,
    alpha_hist=0.5,
    save_path=FIGURES_DIR / f'human_led_collaboration_{identi_slug}_synergy.svg',
    scatter_plot=True,
    # pair_test_fun=False
)


# In[1-5]
'''plot initial and final accuracy for student vs doctor'''

import pandas as pd
import numpy as np

# iorf='initial_correct'
iorf='final_correct'

if iorf=='initial_correct':
    iorf_name='Initial'
    y_range=(0, 0.6)
    color_list=[s_col[0],d_col[0],s_col[1],d_col[1],s_col[2],d_col[2]]

else:
    iorf_name='Final'
    y_range=(0, 0.8)
    color_list=[s_col[0],d_col[0],s_col[1],d_col[1],s_col[2],d_col[2]]

# Load the required data.
df = pd.read_excel(FILES_DIR / 'participant_accuracy_by_condition.xlsx')
df.columns = df.columns.str.strip()

# Analysis step.
g1_med = df[(df['participant_group'] == 'medical_student') & (df['confidence_condition'] == 1)][iorf]
g1_doc = df[(df['participant_group'] == 'ophthalmologist')   & (df['confidence_condition'] == 1)][iorf]
g1_ai = df[(df['participant_group'] == 'ophthalmologist')   & (df['confidence_condition'] == 1)]['ai_correct']
g2_med = df[(df['participant_group'] == 'medical_student') & (df['confidence_condition'] == 2)][iorf]
g2_doc = df[(df['participant_group'] == 'ophthalmologist')   & (df['confidence_condition'] == 2)][iorf]
g2_ai = df[(df['participant_group'] == 'ophthalmologist')   & (df['confidence_condition'] == 2)]['ai_correct']
g3_med = df[(df['participant_group'] == 'medical_student') & (df['confidence_condition'] == 3)][iorf]
g3_doc = df[(df['participant_group'] == 'ophthalmologist')   & (df['confidence_condition'] == 3)][iorf]
g3_ai= df[(df['participant_group'] == 'ophthalmologist')   & (df['confidence_condition'] == 3)]['ai_correct']


# Analysis step.
max_n = max(len(g1_med), len(g1_doc),len(g1_ai), len(g2_med), 
            len(g2_doc),len(g2_ai), len(g3_med), len(g3_doc),len(g3_ai))

def pad_with_nan(series, target_len):
    """Perform the documented analysis step."""
    if len(series) >= target_len:
        return series.iloc[:target_len].values
    else:
        pad = np.full(target_len - len(series), np.nan)
        return np.concatenate([series.values, pad])

# Analysis step.
col1 = pad_with_nan(g1_med, max_n)   # Analysis step.
col2 = pad_with_nan(g1_doc, max_n)   # Analysis step.
# Analysis step.
col3 = pad_with_nan(g2_med, max_n)   # Analysis step.
col4 = pad_with_nan(g2_doc, max_n)   # Analysis step.
# Analysis step.
col5 = pad_with_nan(g3_med, max_n)   # Analysis step.
col6 = pad_with_nan(g3_doc, max_n)   # Analysis step.
# Analysis step.

# Analysis step.
result = np.column_stack([col1, col2, col3, col4, col5, col6])

# Save the generated result.

# Create the analysis plot.
# final_data = result
plot_bar(
    data=result,
    figsize_cm=(10, 6),
    ylim=y_range,
    line_plot=[],
    large_gap_cols=[(1,2),(3,4)],
    col_names=['stu', 'doc','stu', 'doc','stu', 'doc'],
    ylabel=f'{iorf_name} accuracy',
    title='',
    bar_colors= color_list,
    sig_test_vs_zero=[],
    paired_tests=[(0,1),(2,3),(4,5)],
    errorbar='sem',  # Plot or analysis configuration.
    alpha_scatter=1,
    alpha_hist=0.5,
    save_path=FIGURES_DIR / f'{iorf_name.lower()}_accuracy_medical_student_vs_ophthalmologist.svg',
    scatter_plot=True,
    pair_test_fun=False
)

# In[4-0]
'''bayesian fit and simulation and params save: bayesian_correct/bayesian_change_or_stay'''
import pandas as pd
from statsmodels.formula.api import logit
import numpy as np

# Save the generated result.
results_df = pd.DataFrame(columns=[
    'participant_id', 'participant_group', 'confidence_condition',
    'beta0', 'beta1', 'beta2',
    'initial_log_odds_p_value', 'ai_log_odds_p_value',
    'initial_log_odds_significance', 'ai_log_odds_significance'
])

# Load the required data.
filepath = DATA_DIR / 'retinal_diagnosis_trials.xlsx'
df_original = pd.read_excel(filepath)

# df_fit = df_original[df_original['ai_initial_agreement'] == 0].copy()
df_fit = df_original[
    (df_original['ai_initial_agreement'] == 0) &
    ~((df_original['ai_correct'] == 0) & (df_original['initial_correct'] == 0))
].copy()
# # print(f"original: {len(df_original)}")
# # print(f"AI human choice incong: { (df_original['ai_initial_agreement'] == 0).sum() }")
# # print(f"AI human choice incong and corr incong: {len(df_fit)}")
df_original['bayesian_change_or_stay'] = np.nan
df_original['bayesian_correct'] = np.nan   # Analysis step.
# Analysis step.
# Aggregate participant-level results.

for subj_id in df_fit['participant_id'].unique():
    # Aggregate participant-level results.
    
    identity = df_fit[df_fit['participant_id'] == subj_id]['participant_group'].iloc[0]  # Analysis step.

    # Analysis step.
    data2 = df_fit[(df_fit['participant_id'] == subj_id) & (df_fit['confidence_condition'] == 2)]
    if len(data2) >= 3:
        try:
            model2 = logit('ai_correct ~ initial_confidence_log_odds + ai_confidence_log_odds', data=data2).fit_regularized(
                method='l1',          # L1 regularization (Lasso)
                alpha=0.1,            # Regularization strength
                trim_mode='auto',
                maxiter=2000,
                disp=True
            )
            # model2 = logit('ai_correct ~ initial_confidence_log_odds + ai_confidence_log_odds', data=data2).fit(disp=0)
            
            # Analysis step.
            beta0_2 = model2.params['Intercept']
            beta1_2 = model2.params['initial_confidence_log_odds']
            beta2_2 = model2.params['ai_confidence_log_odds']
            p1_2 = model2.pvalues['initial_confidence_log_odds']
            p2_2 = model2.pvalues['ai_confidence_log_odds']
            sig1_2 = 'significant' if p1_2 < 0.05 else 'not_significant'
            sig2_2 = 'significant' if p2_2 < 0.05 else 'not_significant'

            # Save the generated result.
            results_df = pd.concat([results_df, pd.DataFrame([{
                'participant_id': subj_id,
                'participant_group': identity,
                'confidence_condition': 2,
                'beta0': beta0_2,
                'beta1': beta1_2,
                'beta2': beta2_2,
                'initial_log_odds_p_value': p1_2,
                'ai_log_odds_p_value': p2_2,
                'initial_log_odds_significance': sig1_2,
                'ai_log_odds_significance': sig2_2
            }])], ignore_index=True)
            # Analysis step.
            rows_to_predict2 = (df_original['confidence_condition'] == 2) & (df_original['participant_id'] == subj_id)
            pred2 = model2.predict(df_original.loc[rows_to_predict2, ['initial_confidence_log_odds', 'ai_confidence_log_odds']])
            change_or_stay2 = (pred2 > 0.5).astype(int)
            df_original.loc[rows_to_predict2, 'bayesian_change_or_stay'] = change_or_stay2

            # Analysis step.
            df_original.loc[rows_to_predict2 & (change_or_stay2 == 1), 'bayesian_correct'] = \
                df_original.loc[rows_to_predict2 & (change_or_stay2 == 1), 'ai_correct'].values
            
            df_original.loc[rows_to_predict2 & (change_or_stay2 == 0), 'bayesian_correct'] = \
                df_original.loc[rows_to_predict2 & (change_or_stay2 == 0), 'initial_correct'].values
            
            
            # Analysis step.
            # rows_to_predict2 = df_original['confidence_condition'] == 2
            # pred2 = model2.predict(df_original.loc[rows_to_predict2, ['initial_confidence_log_odds', 'ai_confidence_log_odds']])
            # df_original.loc[rows_to_predict2, 'bayesian_change_or_stay'] = (pred2 > 0.5).astype(int)
            
            # Analysis step.
        except:
            print("Condition 2 model fitting failed", end=' ')
    else:
        print("Insufficient data for condition 2", end=' ')

    # Analysis step.
    data3 = df_fit[(df_fit['participant_id'] == subj_id) & (df_fit['confidence_condition'] == 3)]
    if len(data3) >= 3:
        try:
            model3 = logit('ai_correct ~ initial_confidence_log_odds + ai_confidence_log_odds', data=data3).fit_regularized(
                method='l1',          # L1 regularization (Lasso)
                alpha=0.1,            # Regularization strength
                trim_mode='auto',
                maxiter=2000,
                disp=True
            )
            # model3 = logit('ai_correct ~ initial_confidence_log_odds + ai_confidence_log_odds', data=data3).fit(disp=0)
            
            # Analysis step.
            beta0_3 = model3.params['Intercept']
            beta1_3 = model3.params['initial_confidence_log_odds']
            beta2_3 = model3.params['ai_confidence_log_odds']
            p1_3 = model3.pvalues['initial_confidence_log_odds']
            p2_3 = model3.pvalues['ai_confidence_log_odds']
            sig1_3 = 'significant' if p1_3 < 0.05 else 'not_significant'
            sig2_3 = 'significant' if p2_3 < 0.05 else 'not_significant'

            # Save the generated result.
            results_df = pd.concat([results_df, pd.DataFrame([{
                'participant_id': subj_id,
                'participant_group': identity,
                'confidence_condition': 3,
                'beta0': beta0_3,
                'beta1': beta1_3,
                'beta2': beta2_3,
                'initial_log_odds_p_value': p1_3,
                'ai_log_odds_p_value': p2_3,
                'initial_log_odds_significance': sig1_3,
                'ai_log_odds_significance': sig2_3
            }])], ignore_index=True)

            # Analysis step.
            rows_to_predict3 = (df_original['confidence_condition'] == 3) & (df_original['participant_id'] == subj_id)
            pred3 = model3.predict(df_original.loc[rows_to_predict3, ['initial_confidence_log_odds', 'ai_confidence_log_odds']])
            change_or_stay3 = (pred3 > 0.5).astype(int)
            df_original.loc[rows_to_predict3, 'bayesian_change_or_stay'] = change_or_stay3

            # Analysis step.
            df_original.loc[rows_to_predict3 & (change_or_stay3 == 1), 'bayesian_correct'] = \
                df_original.loc[rows_to_predict3 & (change_or_stay3 == 1), 'ai_correct'].values
            
            df_original.loc[rows_to_predict3 & (change_or_stay3 == 0), 'bayesian_correct'] = \
                df_original.loc[rows_to_predict3 & (change_or_stay3 == 0), 'initial_correct'].values
            # Analysis step.
            # rows_to_predict3 = df_original['confidence_condition'] == 3
            # pred3 = model3.predict(df_original.loc[rows_to_predict3, ['initial_confidence_log_odds', 'ai_confidence_log_odds']])
            # df_original.loc[rows_to_predict3, 'bayesian_change_or_stay'] = (pred3 > 0.5).astype(int)
            
            # Analysis step.
        except:
            print("Condition 3 model fitting failed")
    else:
        print("Insufficient data for condition 3")

# Save the generated result.
df_original.to_excel(FILES_DIR / 'retinal_diagnosis_trials_baye.xlsx', index=False)


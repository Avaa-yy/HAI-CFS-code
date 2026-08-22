# HAI-CFS retinal diagnosis analysis

This repository contains the retinal disease diagnosis data and Python analysis code for evaluating conventional human-led AI collaboration and the HAI-CFS framework. 

## Repository structure

- `data/retinal_diagnosis_trials.xlsx`: trial-level retinal diagnosis data used by the analyses.
- `data/data_dictionary.csv`: definitions of the English data fields.
- `scripts/human_led_collaboration_analysis.py`: diagnostic accuracy, synergy, participant-group comparisons, and Bayesian analysis for conventional human-led AI collaboration.
- `scripts/hai_cfs_analysis.py`: HAI-CFS delegation and integration, confidence-threshold fitting, and stability analysis.
- `plotting/`: shared plotting functions.
- `files/`: intermediate Excel workbooks generated during analysis, including the per-seed results.
- `figures/`: generated SVG figures.

## Environment

Python 3.11 is recommended.

Using conda:

```bash
conda env create -f environment.yml
conda activate hai-cfs
```

Using pip:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Run the analyses

Run commands from the repository root.

Analyze conventional human-led AI collaboration:

```bash
python scripts/human_led_collaboration_analysis.py
```

Analyze HAI-CFS strategies and random-split stability:

```bash
python scripts/hai_cfs_analysis.py
```

The HAI-CFS stability workflow uses seeds 42 through 141. Statistical significance is displayed in the generated figures as `ns`, `*`, `**`, or `***`.

## Tests

```bash
pytest -q
```

## Large Excel files

Excel files are configured for Git LFS because the stability workflow can generate many binary workbooks:

```bash
git lfs install
git lfs pull
```

## Data fields

The input workbook and all generated workbooks use English field names and categorical labels. Column definitions are provided in `data/data_dictionary.csv`.

## License

This repository is released under the MIT License.

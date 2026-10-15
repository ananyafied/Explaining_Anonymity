# Explaining_Anonymity
Comparing the interpretability of black-box models used for anonymous fake account detection, bot and troll behavior analysis in social media spaces. Final project for Dr Brinnae Bent's class on Explainable AI (XAI) at Duke University.

## Status
1. **Synthetic data with known ground truth** (done): `anonsynth/`, documented in [docs/synthetic_data.md](docs/synthetic_data.md)
2. Fit and evaluate 28 models on the synthetic data: [notebooks/01_fit_models.ipynb](notebooks/01_fit_models.ipynb) (written, **not yet run**; under review)
3. Explainability methods applied to the stored models (next)

## Quick start
```bash
pip install -r requirements-dev.txt
python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic   # writes data/synthetic/
python -m pytest                                                       # checks the generator
```

## Fitting the models (step 1)
```bash
pip install -r requirements-dev.txt -r requirements-models.txt   # PyTorch Geometric: see the note in requirements-models.txt
python -m anonsynth --out data/synthetic
jupyter lab notebooks/01_fit_models.ipynb
```
Outputs (models, predictions, metrics) go to `outputs/step1_fit/`, which is gitignored.

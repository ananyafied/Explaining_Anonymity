# Explaining_Anonymity
Comparing the interpretability of black-box models used for anonymous fake account detection, bot and troll behavior analysis in social media spaces. Final project for Dr Brinnae Bent's class on Explainable AI (XAI) at Duke University.

## Status
1. **Synthetic data with known ground truth** (done): `anonsynth/`, documented in [docs/synthetic_data.md](docs/synthetic_data.md)
2. Model fitting and explainability evaluation (next)

## Quick start
```bash
pip install -r requirements-dev.txt
python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic   # writes data/synthetic/
python -m pytest                                                       # checks the generator
```

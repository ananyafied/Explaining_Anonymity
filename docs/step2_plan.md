# Step 2 plan: explaining the 28 fitted models

**Status:** draft v1 (2026-10-15), to be revised. No step-2 code has been written.

Step 1 fitted 28 models (plus a baseline) on the synthetic data, and the results are in `outputs/step1_fit/`. Step 2 applies explainability methods to those models and compares how explainable each one is. Accuracy is not the focus.

## Decisions recorded
- The known issues from step 1 are **not** being fixed for now: the belief-propagation model's inverted scores, the duplicate SVD model (identical to PCA), the weak models (dense autoencoder, label spreading), and the data confounds (node degree in the graph, post volume in the sequences, the decoy hashtag in the text).
- Each model gets **two** explainability methods, the two most commonly used for its model type.
- Four evaluation criteria (below).
- One notebook per model family, written first and reviewed, then run one cell at a time on the user's machine.

## 1. Two methods per model

| Models | Method 1 | Method 2 |
|---|---|---|
| **Tabular, supervised**: random forest, XGBoost, LightGBM, RBF SVM, linear SVM, PCA / NMF / SVD + logistic regression | SHAP | LIME |
| **Tabular, anomaly**: Isolation Forest, LOF, one-class SVM, dense autoencoder | SHAP on the anomaly score | LIME on the anomaly score |
| **Sequence, CNN** | Grad-CAM | Integrated Gradients |
| **Sequence, RNN, LSTM, LSTM autoencoder** | Integrated Gradients | GradientSHAP |
| **Text**: transformer, word2vec + LR, doc2vec + LR | SHAP (text) | LIME (text) |
| **Graph neural nets**: GCN, GAT, GraphSAGE, R-GCN | GNNExplainer | Integrated Gradients on node features |
| **Graph embeddings**: node2vec, DeepWalk + LR | SHAP | LIME |
| **Multimodal fusion** | SHAP over the three input blocks (text, profile stats, graph) | Drop-one-modality ablation |
| **Label spreading, belief propagation** | Which seed labels and neighbours drove the score | Remove the top contributor and re-score |

Notes:
- Grad-CAM only applies to the CNN, because it needs convolutional layers.
- Label spreading and belief propagation are not standard ML models, so SHAP and LIME do not apply directly. They get intrinsic and perturbation-based explanations instead.
- Dropping PDP, ALE and counterfactuals means there is no check that an explainer recovers the *shape* of the nonlinear threshold effect.
- Libraries needed: `shap`, `lime`, `captum`, and PyTorch Geometric's explain module (GNNExplainer and its Captum wrapper). Installs are untested.

## 2. What gets measured

The synthetic data comes with ground truth (`truth/`), so explanations can be scored, not just inspected.

1. **Correctness:** does the explanation find the planted signal (the true formula, burst hours, marker words, ring members)?
2. **Faithfulness:** does removing the top-attributed inputs actually change the prediction?
3. **Agreement:** do the model's two methods rank inputs the same way?
4. **Stability:** do repeated runs give the same answer? This matters most for the random methods (LIME, KernelSHAP).

Runtime is logged for free.

## 3. Comparison
- **Primary:** the 12 models that read the same 12 profile stats.
- **Secondary:** within sequences, text and graph.
- Weak models (belief propagation, label spreading, dense autoencoder) are explained anyway, but flagged as low-signal in the write-up.
- PCA and SVD are the same model, so their explanations should match, which is a free sanity check of the pipeline.
- Degree, volume and the decoy hashtag are treated as extra known shortcuts, and we check whether the explainers reveal them.
- Local methods run on a fixed sample of test accounts (about 50 each of human, bot and troll), the same accounts for every model. Global methods use the full test set. Background data comes from train.

## 4. Order of work
1. Tabular (the true formula gives the sharpest ground truth).
2. Sequences.
3. Text.
4. Graph and multimodal.
5. Cross-model comparison and write-up.

One notebook per family, with a shared helper module and a common results format saved to `outputs/step2_explain/`.

## 5. Pitfalls to design around
- **Correlated features:** `posts_per_week` is a noisy copy of `posts_per_day`, so credit gets split. SHAP and LIME will handle this differently.
- **LIME and KernelSHAP are random and slow:** budget re-runs for the stability checks.
- **Near-chance and flawed models** cannot be meaningfully explained, so read their results with that in mind.

# Step 2 plan: explaining the 28 fitted models

**Status:** draft v2 (2026-10-15). Notebooks are written on the `step2-explainability` branch and have not yet been run end to end by the project owner.

Step 1 fitted 28 models (plus a baseline) on the synthetic data; results are in `outputs/step1_fit/`. Step 2 applies explainability methods to those models and compares how explainable each one is. Accuracy is not the focus.

## Decisions recorded
- The known issues from step 1 are **not** being fixed: the belief-propagation model's inverted scores, the duplicate SVD model (identical to PCA), the weak models (dense autoencoder, label spreading), and the data confounds (node degree in the graph, post volume in the sequences, the decoy hashtag in the text). They are treated as known shortcuts and flagged in results.
- Each model gets **two** explainability methods, the two most commonly used for its model type (section 1).
- Four evaluation criteria (section 2).
- Models are compared **within groups defined by input type**, not across groups (section 3).
- One notebook per family, written first and reviewed, then run one cell at a time on the owner's machine (section 4).

## 1. Two methods per model

| Models | Method 1 | Method 2 |
|---|---|---|
| **Tabular, supervised**: random forest, XGBoost, LightGBM, RBF SVM, linear SVM, PCA / NMF / SVD + logistic regression | SHAP | LIME |
| **Tabular, anomaly**: Isolation Forest, LOF, one-class SVM, dense autoencoder | SHAP on the anomaly score | LIME on the anomaly score |
| **Sequence, CNN** | Grad-CAM | Integrated Gradients |
| **Sequence, RNN, LSTM, LSTM autoencoder** | Integrated Gradients | GradientSHAP |
| **Text**: transformer, word2vec + LR, doc2vec + LR | SHAP (text) | LIME (text) |
| **Graph neural nets**: GCN, GAT, GraphSAGE, R-GCN | GNNExplainer | Integrated Gradients (node features and edges) |
| **Graph embeddings**: node2vec, DeepWalk + LR | SHAP | LIME |
| **Multimodal fusion** | Shapley values over the three input blocks (text, profile stats, graph) | Drop-one-modality ablation |
| **Label spreading, belief propagation** | Which seed labels and neighbours drove the score | Remove the top contributor and re-score |

Notes:
- Grad-CAM only applies to the CNN, because it needs convolutional layers.
- Label spreading and belief propagation are not standard ML models, so SHAP and LIME do not apply directly. They get intrinsic and perturbation-based explanations instead.
- Dropping PDP, ALE and counterfactuals means there is no check that an explainer recovers the *shape* of the nonlinear threshold effect.
- Libraries: `shap`, `lime`, `captum`, and PyTorch Geometric's explain module (GNNExplainer and its Captum wrapper).

## 2. What gets measured, and how to read it

The synthetic data comes with ground truth (`truth/`), so explanations can be scored, not just inspected. Four criteria, each defined concretely per group in section 3:

1. **Correctness:** does the explanation find the planted signal?
2. **Faithfulness:** does removing the top-attributed inputs change the prediction more than removing random inputs? (Change in score, divided by the standard deviation of the score so that probabilities and anomaly scores are comparable.)
3. **Agreement:** do the model's two methods rank inputs the same way?
4. **Stability:** does a small random perturbation of the input leave the explanation (nearly) unchanged?

Runtime (seconds per explanation) is logged for free.

These criteria judge the *explanations*, not the models directly: a model counts as more explainable when its explanations score well. They are measured explainability, not human understandability; there is no user study.

**Reading the patterns**

| Pattern | Interpretation |
|---|---|
| High correctness and high faithfulness | The explainer is trustworthy, and the model uses the real signal. |
| High faithfulness, low correctness | The explainer is right about the model, but the model relies on a shortcut (the decoy hashtag, node degree, post volume, a proxy feature). |
| High agreement, low correctness | Both methods share a blind spot, for example crediting the proxy `posts_per_week`. |
| Low stability | The explanation cannot be trusted, whatever its other scores. |
| Low faithfulness | The explanation does not describe what the model actually does. |

## 3. Comparison: within groups by input type

| Group | Models | Unit of explanation | Ground truth |
|---|---|---|---|
| **Profile stats (12 models)** | random forest, XGBoost, LightGBM, RBF SVM, linear SVM, PCA / NMF / SVD + LR, Isolation Forest, LOF, one-class SVM, dense autoencoder | the 12 features | true per-feature contributions to the label (`truth/accounts_truth.csv`, `term_*` columns) |
| **Posting timeline (4)** | CNN, RNN, LSTM, LSTM autoencoder | the 168 hours | planted burst hours (`truth/sequence_masks.npz`) |
| **Post text (3)** | transformer, word2vec + LR, doc2vec + LR | the words of a post | planted marker words (`truth/posts_truth.csv`); the decoy hashtag is tracked separately as a shortcut |
| **Graph (8)** | GCN, GAT, GraphSAGE, R-GCN; node2vec, DeepWalk, label spreading, belief propagation | edges, neighbours, seeds, node features | ring edges and ring members (`truth/edges_truth.csv`, `ring_id`) |
| **Multimodal (1)** | fusion model | the three input blocks | none for block weights; compared only with the single-modality models it is built from |

The graph group splits into two sub-groups: the four GNNs (which also read profile stats) and the four structure-only models.

**How the criteria are defined per group**

- *Profile stats.* Correctness: rank correlation between mean |attribution| and the true per-feature importance; share of attribution on the six truly informative features (versus decoys, noise and the proxy); top-5 precision. For the four anomaly detectors the score is not trained on the label, so correctness is only partly meaningful and the other three criteria carry more weight. Faithfulness: replace the top-attributed features with the training mean and measure the score change versus random features. Stability: add small Gaussian noise (5% of each feature's standard deviation) and correlate the attributions.
- *Timeline.* Correctness: AUC of |attribution| against the planted-burst mask (bots and trolls only), and the share of attribution mass inside the mask. Faithfulness: zero out the top-attributed hours. A volume probe permutes the hours of each series, which keeps total volume and destroys timing, to test whether the model uses timing or just volume.
- *Text.* Correctness: rank of the planted marker word among the post's words; share of attribution on marker words. Shortcut detection: attribution given to the decoy hashtag. Faithfulness: delete the top-attributed words. Stability: delete one random filler word and re-explain.
- *Graph.* Correctness (GNNs): AUC of edge importance against ring edges for troll nodes. Faithfulness: remove the top-attributed edges. A degree probe removes half of a node's edges at random to test reliance on node degree. Label spreading and belief propagation: do the top-contributing seeds belong to the same ring as the explained troll? Structure-only embedding models: correctness is not defined at the level of embedding dimensions, so only the other three criteria are reported, plus a probe of how well the embedding predicts degree and ring membership.
- *Multimodal.* Block-level Shapley shares by account kind, drop-one-modality effects, faithfulness and stability. No correctness score.

**Reporting.** One scorecard per group, with bootstrap intervals over the explained instances. Models are ranked per criterion; there is no single composite number. With three or four models per group there are no formal significance tests, only intervals. Cross-group comparison stays qualitative, because the tasks differ in difficulty (the timelines are near-perfectly separable and the profile stats are not).

**Sample.** Local methods run on a fixed sample of test accounts (50 each of human, bot and troll), the same accounts for every model; the text notebook uses a fixed sample of posts. Background data comes from the training split.

## 4. Notebooks and layout

| Notebook | Models |
|---|---|
| `notebooks/02_explain_tabular.ipynb` | 12 profile-stats models |
| `notebooks/03_explain_sequences.ipynb` | CNN, RNN, LSTM, LSTM autoencoder |
| `notebooks/04_explain_text.ipynb` | transformer, word2vec, doc2vec |
| `notebooks/05_explain_graph.ipynb` | 4 GNNs, node2vec, DeepWalk, label spreading, belief propagation |
| `notebooks/06_explain_multimodal.ipynb` | fusion model |
| `notebooks/07_scorecards.ipynb` | reads the results of 02 to 06; builds the per-group scorecards |

Shared code lives in `xai_eval/` (data views and model reloading that mirror step 1, model class definitions, metrics). Results go to `outputs/step2_explain/<family>/`. Every notebook first reloads the step-1 models and checks that they reproduce the stored step-1 predictions before explaining anything. Install extra libraries with `requirements-xai.txt`.

## 5. Pitfalls to design around
- **Correlated features:** `posts_per_week` is a noisy copy of `posts_per_day`, so credit gets split. SHAP and LIME will handle this differently.
- **LIME and KernelSHAP are random and slow:** sample sizes are kept modest, and these methods are seeded.
- **Near-chance and flawed models** cannot be meaningfully explained, so read their results with that in mind.
- **Reloading models:** neural-network weights are saved without their class definitions, so `xai_eval/step1_models.py` must stay identical to the model definitions in the step-1 notebook.

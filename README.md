# LLM Classification Fine-tuning

Predict user preferences in [Chatbot Arena](https://chat.lmsys.org/) head-to-head LLM battles. Given two anonymous LLM responses to the same prompt, this project builds a fine-tuned classification model to predict which response a user will prefer — or if they'll call it a tie.

## Motivation

Direct LLM-based preference prediction suffers from well-documented biases:

- **Position bias** — favoring whichever response is presented first
- **Verbosity bias** — favoring longer responses regardless of quality
- **Self-enhancement bias** — LLMs favoring responses similar to their own outputs

This project trains a supervised model that captures genuine quality signals rather than relying on these superficial heuristics.

## Approach

The full pipeline lives in a single notebook (`notebooks/llm-classification-model.ipynb`) and follows these stages:

1. **Data Loading & Parsing** — Load Chatbot Arena conversation data; parse multi-turn JSON prompt/response fields
2. **Exploratory Data Analysis** — Class distribution, model win rates, text length distributions, position bias analysis
3. **Feature Engineering**
   - Hand-crafted features: response length, readability scores, structural markers (code blocks, lists, headers), hedging/confidence language, example usage
   - Sentence embeddings via `all-MiniLM-L6-v2` with PCA dimensionality reduction
   - Cosine similarity between prompt and response embeddings
   - Bias mitigation features (e.g., length difference normalization)
4. **Model Development**
   - Logistic Regression baseline
   - LightGBM and XGBoost with Optuna hyperparameter tuning (50 trials)
   - Stratified 5-fold cross-validation
5. **Ensemble** — Optimized weighted average of tuned LightGBM and XGBoost models
6. **Evaluation** — Confusion matrices, SHAP feature importance, performance dashboard
7. **Submission** — Probability predictions for three outcome classes

## Project Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv                       # Labeled training data
│   ├── test.csv                        # Unlabeled test data
│   └── sample_submission.csv           # Expected submission format
├── configs/
│   ├── experiment.yaml                 # Experiment name, seed, debug settings
│   ├── features.yaml                   # Embedding model, PCA dims, feature toggles
│   ├── models.yaml                     # LightGBM/XGBoost/Logistic hyperparameters
│   └── paths.yaml                      # Input/output directory paths
├── notebooks/
│   └── llm-classification-model.ipynb  # Full implementation
└── outputs/
    ├── logs/                           # Training logs
    ├── models/                         # Saved model artifacts
    ├── predictions/                    # Test predictions & submissions
    └── reports/                        # Evaluation reports
```

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd llm-classification-finetuning

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (used for readability features)
python -c "import nltk; nltk.download('punkt')"
```

Place the dataset files (`train.csv`, `test.csv`, `sample_submission.csv`) in the `data/` directory.

## Usage

Open and run the notebook:

```bash
jupyter notebook notebooks/llm-classification-model.ipynb
```

The notebook is self-contained — it loads configs from `configs/`, reads data from `data/`, and writes all outputs to `outputs/`.

To run in debug mode with a smaller data sample, set `debug: true` in `configs/experiment.yaml`.

## Configuration

All hyperparameters and settings are externalized in YAML configs under `configs/`:

| File | Controls |
|------|----------|
| `experiment.yaml` | Experiment name, random seed, debug mode |
| `features.yaml` | Embedding model, PCA dimensions, feature toggles, NLP word lists |
| `models.yaml` | Model hyperparameters, Optuna search spaces, ensemble method |
| `paths.yaml` | Data and output directory paths |

## Evaluation

The primary metric is **log loss** across three outcome classes (`winner_model_a`, `winner_model_b`, `winner_tie`). The notebook compares model performance against a uniform baseline and reports per-fold cross-validation scores.

## Tech Stack

- **ML**: scikit-learn, LightGBM, XGBoost, Optuna
- **NLP**: Hugging Face Transformers, Sentence-Transformers, NLTK
- **Data**: pandas, NumPy
- **Visualization**: matplotlib, seaborn, plotly, SHAP

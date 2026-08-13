# When AI Rewrites, Classifiers Relax

Code and data for the paper: "When AI Rewrites, Classifiers Relax: Uncertainty-Aware Sentiment Analysis on Sarcastic and AI-Paraphrased Social Text"


## Pipeline Overview

Two notebooks run in sequence. Run the CPU notebook first, then the GPU notebook.

**Notebook 1: sentiment-cpu.ipynb**
- Phase 1: Baseline sentiment evaluation on iSarcasm (VADER + RoBERTa)
- Phase 3 (partial): Calibration on clear-polarity Yelp subset, sarcasm instability stats

**Notebook 2: sentiment-gpu-2.ipynb**
- Phase 2: AI paraphrase generation using Qwen3.5-4B and Gemma4-E4B via Ollama
- Phase 2.6/2.7: Sentiment classification on original vs paraphrased Yelp reviews
- Phase 3.3: ECE on AI-paraphrased text
- Phase 3.5: Semantic Entropy vs MC-Dropout AUROC comparison on iSarcasm
- Phase 4: Abstention wrapper (confidence threshold 0.6)
- Phase 5: Explanation generation for flagged ambiguous inputs

## Requirements

**CPU notebook:** Python 3.10+, transformers, datasets, vaderSentiment, scipy, scikit-learn, pandas, numpy

**GPU notebook:** All of the above plus Ollama running locally with Qwen3.5-4B and Gemma4-E4B pulled. Run on Kaggle T4x2 GPU or equivalent.

Install Ollama models:
```
ollama pull qwen3.5:4b
ollama pull gemma4:e4b
```

## Datasets

Download from HuggingFace before running:

- iSarcasm: load_dataset("iSarcasm")
- Yelp Polarity: load_dataset("yelp_polarity")
- HC3: load_dataset("Hello-SimpleAI/HC3")

Raw data files are not included in this repository due to licensing. The notebooks handle downloading automatically via the datasets library.

## Result Files

Pre-computed results are included for reproducibility:

- baseline_results.json: Phase 1 outputs
- drift_results.json: AI-text accuracy and semantic drift
- stratified_ece_final.json: ECE across content types
- uncertainty_method_comparison.json: Semantic Entropy vs MC-Dropout AUROC
- abstention_ablation.json: Abstention wrapper accuracy improvement
- phase5_explanations_full.csv: Natural language explanations for 700 flagged examples

## Compute

All experiments run on Kaggle free tier (T4x2 GPU for GPU notebook, CPU runtime for CPU notebook). No paid compute required.
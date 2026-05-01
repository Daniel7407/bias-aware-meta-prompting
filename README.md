# Sensitivity Analysis of Emotion Classification via Prompt Ordering

This repository contains an experimental framework to evaluate how the sequence of few-shot examples influences the classification performance of local Large Language Models (LLMs). The study specifically examines the sensitivity of model outputs to example permutations using the TweetEval emotion dataset.

## Experimental Overview
The primary objective is to determine the extent to which "ordering matters" in in-context learning. By providing the same set of few-shot examples in different sequences, we measure the variance in model predictions. High variance indicates a lack of robust task understanding and a high dependency on prompt structure.

## Technical Specifications
- **Models Evaluated:** Qwen 3.5 (0.8B) and Llama 3.2 (1B).
- **Inference Engine:** Ollama (AsyncClient).
- **Dataset:** TweetEval (Emotion subset), focusing on Anger, Joy, Optimism, and Sadness.
- **Evaluation Metric:** Pairwise Disagreement Rate and Accuracy per Permutation.

## Prerequisites
Ensure Ollama is installed and the following models are available locally:
- `qwen3.5:0.8b`
- `llama3.2:1b`

Required Python libraries:
- `ollama`
- `datasets`
- `numpy`
- `tqdm`
- `asyncio`

## Implementation Modes
The script supports four experimental configurations:
1. **Zero-Shot:** Classification without external examples to establish a baseline.
2. **Few-Shot:** Standard few-shot prompting with four distinct example permutations.
3. **Few-Shot Ranking:** A multi-stage prompt requiring the model to assign scores to all labels before finalizing a classification.
4. **Few-Shot Bias-Aware:** An optimized prompt designed to instruct the model to ignore example ordering.

## Execution Instructions
1. Configure the environment variable for parallel processing if required:
   `export OLLAMA_NUM_PARALLEL=2`
2. Select the experiment mode in the `MAIN EXECUTION` section of the script.
3. Execute the script:
   `python bias_aware_classifier.py`

## Data Analysis
The output provides a statistical breakdown of:
- **Total Pairwise Disagreement Rate:** The frequency with which the model changes its prediction on a single test sample when the order of examples is shuffled.
- **Accuracy by Sequence:** Performance metrics for each specific permutation.
- **Label Distribution:** A tracking of whether the model shows a preference for the label of the first example in the sequence.

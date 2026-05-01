import os
import random
import numpy as np
import asyncio
from datasets import load_dataset
from tqdm import tqdm
from ollama import AsyncClient
from collections import Counter
import itertools

os.environ["OLLAMA_NUM_PARALLEL"] = "2"

# Set seeds
random.seed(42)
np.random.seed(42)

dataset = load_dataset("tweet_eval", "emotion")
label_map = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}
target_labels = [0, 1, 2, 3]

all_data = []
for split in ['train', 'validation', 'test']:
    for item in dataset[split]:
        all_data.append(item)

samples_by_label = {label: [] for label in target_labels}
random.shuffle(all_data)
for item in all_data:
    lbl = item['label']
    if lbl in target_labels:
        samples_by_label[lbl].append(item['text'])

FEW_SHOT_POOL = {}
test_samples = []
for lbl_id, lbl_name in label_map.items():
    pool = samples_by_label[lbl_id]
    FEW_SHOT_POOL[lbl_name] = {"text": pool.pop(0), "label": lbl_name}
    for _ in range(50):
        test_samples.append({"text": pool.pop(0), "label": lbl_name})

random.shuffle(test_samples)

A, J, O, S = FEW_SHOT_POOL["anger"], FEW_SHOT_POOL["joy"], FEW_SHOT_POOL["optimism"], FEW_SHOT_POOL["sadness"]
PERMUTATIONS = [[A, J, O, S], [J, O, S, A], [O, S, A, J], [S, A, J, O]]


# --- STEP 2: PROMPT FUNCTIONS ---

def build_zero_shot(test_text):
    return f"Classify the sentiment of the following tweet with either anger, joy, optimism, or sadness. Respond with only the sentiment.\n\nTweet: {test_text}\nResult:"

def build_prompt(examples_ordered, test_text):
    prompt = "Classify the final tweet sentiment with either anger, joy, optimism, or sadness. Respond with only the sentiment.\n\n"
    prompt += "### EXAMPLES ###\n"
    for ex in examples_ordered:
        prompt += f"Tweet: {ex['text']}\nResult: {ex['label']}\n\n"
    prompt += f"### TARGET ###\nTweet: {test_text}\nResult:"
    return prompt

def build_rating_prompt(examples_ordered, test_text):
    prompt = "Task: Evaluate the emotion of a Tweet. You must only output the scores.\n\n### EXAMPLES ###\n"
    for ex in examples_ordered:
        prompt += f"Tweet: {ex['text']}\nResult: {ex['label']}\n---\n"
    prompt += f"\n### TARGET TWEET ###\nTweet: {test_text}\n\n### INSTRUCTIONS ###\n"
    prompt += "Assign a score (0-10) for each emotion based ONLY on the Target Tweet above. Use this exact format:\n"
    labels = [ex['label'] for ex in examples_ordered]
    for label in labels[::-1]:
        prompt += f"{label}: \n"
    return prompt

def build_bias_aware_prompt(examples_ordered, test_text):
    prompt = "### INSTRUCTIONS ###\n"
    prompt += "Classify the TARGET TWEET sentiment as: anger, joy, optimism, or sadness.\n"
    prompt += "WARNING: Do not simply repeat the first label you see. High-quality analysis requires ignoring the order of examples and focusing ONLY on the Target Tweet.\n\n"
    
    prompt += "### REFERENCE EXAMPLES ###\n"
    for ex in examples_ordered:
        prompt += f"Tweet: \"{ex['text']}\" -> Label: {ex['label']}\n"
    
    prompt += f"\n### TARGET TWEET ###\n"
    prompt += f"Tweet: \"{test_text}\"\n\n"
    
    prompt += "### FINAL CHECK ###\n"
    prompt += "1. Is this label based on the text and not the example order? (Yes)\n"
    prompt += "2. Final Classification (one word only):\n"
    prompt += "Result: "
    
    return prompt


# --- STEP 3: PARSERS ---

def extract_sentiment(output_text):
    output_lower = output_text.lower().strip()
    for label in ["anger", "joy", "optimism", "sadness"]:
        if label in output_lower: return label
    return "unknown"

def extract_raw_scores(output_text):
    scores = {label: 0 for label in ["anger", "joy", "optimism", "sadness"]}
    lines = output_text.strip().lower().split('\n')
    for line in lines:
        clean_line = line.replace(':', ' ').replace('-', ' ').strip()
        parts = clean_line.split()
        if len(parts) >= 2 and parts[0] in scores:
            try:
                score_val = int(''.join(filter(str.isdigit, parts[1].split('/')[0])))
                scores[parts[0]] = score_val
            except: continue
    return scores

def extract_rating_sentiment(output_text):
    scores = extract_raw_scores(output_text)
    return max(scores, key=scores.get)

# --- STEP 4: EXECUTION ---

async def query_model_async(prompt, exp_type, bias=None):
    try:
        client = AsyncClient()
        res = await client.generate(
            model="qwen3.5:0.8b", 
            prompt=prompt, 
            think=False,
            options={
                "temperature": 0
            }
        )
        #res = await client.generate(model="llama3.2:1b", prompt=prompt, options={"temperature": 0})
        resp = res['response']

        if exp_type in ["zero-shot", "few-shot"]:
            return extract_sentiment(resp)
        elif exp_type == "few-shot-ranking":
            return extract_rating_sentiment(resp)
        elif exp_type == "few-shot-bias-aware":
            return extract_sentiment(resp)
    except:
        return "unknown"

async def process_sample_async(idx, sample, permutations, exp_type, biases=None):
    test_text = sample['text']

    if exp_type == "zero-shot":
        pred = await query_model_async(build_zero_shot(test_text), exp_type)
        predictions = [pred] * 4
    else:
        tasks = []
        for j, p in enumerate(permutations):
            if exp_type == "few-shot":
                tasks.append(query_model_async(build_prompt(p, test_text), exp_type))
            elif exp_type == "few-shot-ranking":
                tasks.append(query_model_async(build_rating_prompt(p, test_text), exp_type))
            elif exp_type == "few-shot-bias-aware":
                tasks.append(query_model_async(build_bias_aware_prompt(p, test_text), exp_type))

        predictions = await asyncio.gather(*tasks)

    return {"test_idx": idx, "true_label": sample['label'], "predictions": predictions}

async def run_all_samples(samples, permutations, exp_type, biases=None, batch_size=2):
    results = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        tasks = [process_sample_async(i + j, s, permutations, exp_type, biases) for j, s in enumerate(batch)]
        results.extend(await asyncio.gather(*tasks))
        print(f"Batch {i//batch_size + 1} complete...")
    return results

# --- STEP 5: MAIN EXECUTION ---

async def main():
    # SELECT MODE HERE
    # Options: "zero-shot", "few-shot", "few-shot-ranking", "few-shot-bias-aware"
    mode = "few-shot-bias-aware"

    results = await run_all_samples(test_samples, PERMUTATIONS, exp_type=mode)

    # 3. Calculate Accuracy Metrics
    all_pairs_flips = []

    for res in results:
        p = res['predictions']
        # Create all unique index pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        pairs = list(itertools.combinations(range(len(p)), 2))

        # Count how many pairs do not match
        mismatches = sum(1 for i, j in pairs if p[i] != p[j])
        all_pairs_flips.append(mismatches / len(pairs))

    # --- 2. RE-ORGANIZING BY FIRST EMOTION (PRIMACY BIAS) ---
    # Map Permutation index -> The label of the FIRST example in that sequence
    perm_to_first_emo = [PERMUTATIONS[i][0]['label'] for i in range(4)]

    first_emo_stats = []
    labels_to_track = ["anger", "joy", "optimism", "sadness", "unknown"]

    for i in range(4):
        preds_for_this_perm = [res['predictions'][i] for res in results]
        true_labels = [res['true_label'] for res in results]
        first_emo = perm_to_first_emo[i]

        correct = sum(1 for p, t in zip(preds_for_this_perm, true_labels) if p == t)
        acc = correct / len(results)
        dist = Counter(preds_for_this_perm)

        first_emo_stats.append({
            "perm_idx": i,
            "first_emo": first_emo,
            "accuracy": acc,
            "distribution": {lbl: dist.get(lbl, 0) for lbl in labels_to_track}
        })

    # --- PRINT STATISTICS ---
    print("="*60)
    print(f"{'PRIMACY BIAS & PAIRWISE ANALYSIS':^60}")
    print("="*60)
    print(f"Total Pairwise Disagreement Rate: {np.mean(all_pairs_flips):.2%}")
    print(f"Overall Mean Accuracy:            {np.mean([fs['accuracy'] for fs in first_emo_stats]):.2%}")
    print("-" * 60)
    print(f"{'GUESS DISTRIBUTION BY FIRST EXAMPLE':^60}")
    print(f"{'First Example':<15} | {'Acc':<6} | {'Anger':<5} | {'Joy':<5} | {'Opt':<5} | {'Sad':<5}")
    print("-" * 60)
    for fs in first_emo_stats:
        d = fs['distribution']
        print(f"{fs['first_emo'].capitalize():<15} | {fs['accuracy']:.1%} | "
            f"{d['anger']:<5} | {d['joy']:<5} | {d['optimism']:<5} | {d['sadness']:<5}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 21:04:48 2025

@author: danielhier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize_with_GPT_RAG.py
--------------------------

This script uses GPT-4o (via the OpenAI API) to refine normalization of phenotype terms 
by re-ranking candidate matches from BioBERT. It performs the following:

1. Loads a CSV containing candidate matches with cosine similarity scores from BioBERT
2. For each target term, constructs a prompt listing the top N candidate terms and IDs
3. Sends the prompt to GPT-4o, which selects the best HPO match or declares no match
4. Writes the GPT-selected best matches to a CSV file in the results folder

Inputs:
- CSV file: `data/closest_matches_biobert_with_similarities.csv`

  This file must contain:
  - `target_term`: the clinical or biomedical term to normalize
  - `matched_hp_term_1` to `matched_hp_term_20`: top 20 HPO candidate terms retrieved via BioBERT vector similarity
  - `matched_hp_id_1` to `matched_hp_id_20`: corresponding HPO IDs
  - `similarity_1` to `similarity_20`: cosine similarity scores for each candidate term

  N.B.: These candidate matches have been **pre-computed using BioBERT embeddings** and do not require re-calculation during this script run.

Outputs:
- CSV file: `results/best_matches_gpt_4o_with_RAG_20_candidates_1.csv`

  This file contains:
  - `target_term`: the original input term
  - `best_hpo_term`: the term selected by GPT-4o as the most appropriate normalization
  - `hpo_id`: the selected HPO ID or `"HP:0000000"` if no match was appropriate

Configuration:
- `NUM_CANDIDATES` controls how many BioBERT matches are included in the prompt
- `MAX_TEST_CASES` limits how many rows to process (for testing or speed)

Requirements:
- Python packages: openai, pandas, tqdm
- spaCy model installed (if needed elsewhere): `python -m spacy download en_core_web_lg`
- OpenAI API key stored in environment variable `OPENAI_API_KEY`
# Make sure your OpenAI API key is set in the environment:
# export OPENAI_API_KEY="your-api-key"
openai.api_key = os.getenv('OPENAI_API_KEY')

Author: danielhier
"""

import openai
import pandas as pd
import os
import json
from tqdm import tqdm
from pathlib import Path

# === Configurable Parameters ===
NUM_CANDIDATES = 5         # Use 5, 10, or 20 candidate terms in each prompt
MAX_TEST_CASES = 20        # Use None to process the full dataset

# === Set directory paths relative to this script ===
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / 'data'
results_dir = base_dir / 'results'

input_path = data_dir / 'closest_matches_biobert_with_similarities.csv'
output_path = results_dir / 'best_matches_gpt_4o_with_RAG_20_candidates_1.csv'

# === Initialize OpenAI API ===
openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai

# === Function to extract top N candidate matches from BioBERT ===
def find_top_matches(df, num_candidates=5):
    top_matches = []
    for index, row in df.iterrows():
        term = row['target_term']
        matches = []
        for i in range(1, num_candidates + 1):
            matches.append((
                row.get(f'matched_hp_term_{i}', ''),
                row.get(f'matched_hp_id_{i}', ''),
                row.get(f'similarity_{i}', 0)
            ))
        top_matches.append((term, matches))
    return top_matches

# === Function to call GPT-4o and select best match ===
def find_best_match(term, matches):
    prompt = (
        f"You are given a clinical term to normalize to a concept from the Human Phenotype Ontology (HPO).\n"
        f"Term: {term}\n"
        f"Candidate matches:\n"
    )
    for i, (match_term, hpo_id, similarity) in enumerate(matches, 1):
        prompt += f"{i}. {match_term} (HPO ID: {hpo_id}, Similarity: {similarity:.4f})\n"

    prompt += (
        "\nChoose the best match and return it as JSON like:\n"
        '{ "best_match": "term", "hpo_id": "HP:xxxxxxx" }\n\n'
        "If no match is appropriate, return:\n"
        '{ "best_match": "no appropriate match", "hpo_id": "HP:0000000" }'
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in phenotype normalization using the Human Phenotype Ontology."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0
        )
        response_content = response.choices[0].message.content.strip()
        best_match = json.loads(response_content)
    except Exception as e:
        print(f"OpenAI error on term '{term}': {e}")
        best_match = {"best_match": "no appropriate match", "hpo_id": "HP:0000000"}

    return best_match

# === Load input CSV and optionally limit the number of test cases ===
df_closest_matches = pd.read_csv(input_path)

if MAX_TEST_CASES:
    df_closest_matches = df_closest_matches.head(MAX_TEST_CASES)

# === Process all terms ===
top_matches = find_top_matches(df_closest_matches, num_candidates=NUM_CANDIDATES)
results = []

for term, matches in tqdm(top_matches, desc="Finding best matches"):
    best_hpo = find_best_match(term, matches)
    results.append((term, best_hpo['best_match'], best_hpo['hpo_id']))

# === Save results to output CSV ===
results_df = pd.DataFrame(results, columns=['target_term', 'best_hpo_term', 'hpo_id'])
results_df.to_csv(output_path, index=False)

# === Show results ===
print(f"\n✅ Results saved to: {output_path}")
print(results_df.head())
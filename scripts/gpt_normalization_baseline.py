#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 07:49:35 2024
Author: danielhier

Script Name: gpt_normalization_baseline.py

This script performs baseline normalization of phenotype terms using GPT-4o
without any retrieval augmentation. Each term is directly sent to the LLM
for mapping to the best Human Phenotype Ontology (HPO) concept.

Directory Structure:
- Input:  data/well_formed_terms.csv (with column: extracted_term)
- Output: results/hpo_terms_with_normalizations_gpt-4o.csv

Requirements:
- OPENAI_API_KEY must be set in environment
- Python packages: openai, pandas, tqdm
"""

import openai
import pandas as pd
import os
import json
from tqdm import tqdm
from pathlib import Path

# Set up the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai

# Substitute any valid OpenAI model name here
model_name = 'gpt-4o'
num_rows = 10  # Change to increase number of rows normalized

# Directory-agnostic input/output paths
project_root = Path(__file__).resolve().parents[1]
input_file = project_root / "data" / "well_formed_terms.csv"
output_file = project_root / "results" / f"hpo_terms_with_normalizations_{model_name}.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)

# Function to interact with OpenAI API
def find_best_match(term):
    prompt = (
        f"You are given a term to normalize to a concept from the Human Phenotype Ontology and its HPO_ID.\n"
        f"Term: {term}\n"
        "Pick the best one and return the best one in JSON format as follows:\n"
        "{ \"normalized_term\": \"term\", \"normalized_id\": \"HP:xxxxxxx\" }\n"
        "If none of the terms in the HPO seem appropriate, return: \"no appropriate match\", HPO_ID: \"HP:0000000\"."
    )
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert in normalizing medical terms to the Human Phenotype Ontology (HPO)."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        max_tokens=150,
        temperature=0
    )
    
    response_content = response.choices[0].message.content.strip()
    try:
        best_match = json.loads(response_content)
    except json.JSONDecodeError:
        best_match = {"normalized_term": "no appropriate match", "normalized_id": "HP:0000000"}
    
    return best_match

# Load terms
target_df = pd.read_csv(input_file)
target_terms = target_df['extracted_term'].tolist()[:num_rows]

# Normalize terms
results = []
for term in tqdm(target_terms, desc="Finding best matches"):
    best_hpo_term = find_best_match(term)
    results.append((term, best_hpo_term['normalized_term'], best_hpo_term['normalized_id']))

# Save results
results_df = pd.DataFrame(results, columns=['extracted_term', 'normalized_term', 'normalized_id'])
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
print(results_df.head())

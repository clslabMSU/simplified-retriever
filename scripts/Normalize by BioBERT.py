#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 16:06:03 2025

@author: danielhier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 08:29:22 2024

This program identifies the single most similar HPO (Human Phenotype Ontology) term for each target term
based on BioBERT embeddings. See full docstring above.
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pathlib import Path

# Set base directory relative to this script's location
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
results_dir = base_dir / "results"

# Load BioBERT model and tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get the BioBERT embedding for a term
def get_biobert_embedding(term):
    inputs = tokenizer(term, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Load the HPO vectors from the CSV file
hpo_file_path = data_dir / "hpo_biobert_vectors.csv"
hpo_df_with_vectors = pd.read_csv(hpo_file_path)

# Extract the embeddings and other columns
hpo_embeddings = hpo_df_with_vectors.iloc[:, 3:].values  # Assumes embeddings start from 4th column
hpo_terms = hpo_df_with_vectors['hp_term'].tolist()
hpo_ids = hpo_df_with_vectors['hp_id'].tolist()

# Load the target terms from the CSV file
target_terms_file_path = data_dir / "well_formed_terms.csv"
target_df = pd.read_csv(target_terms_file_path)
target_terms = target_df['extracted_term'].tolist()

# Find the closest match for each target term
closest_matches = []

for term in tqdm(target_terms, desc="Finding closest matches"):
    term_vector = get_biobert_embedding(term).reshape(1, -1)
    similarities = cosine_similarity(term_vector, hpo_embeddings).flatten()
    best_match_idx = np.argmax(similarities)
    best_match = (term, hpo_terms[best_match_idx], hpo_ids[best_match_idx], similarities[best_match_idx])
    closest_matches.append(best_match)

# Create a DataFrame for the closest matches
columns = ['extracted_term', 'matched_hp_term', 'matched_hp_id', 'similarity']
closest_matches_df = pd.DataFrame(closest_matches, columns=columns)

# Save the final DataFrame to a CSV file
output_matches_file_path = results_dir / "normalization_by_BioBERT.csv"
closest_matches_df.to_csv(output_matches_file_path, index=False)

# Display the first few rows of the final DataFrame to verify
print(closest_matches_df.head())
print(f"Closest matches saved to {output_matches_file_path}")
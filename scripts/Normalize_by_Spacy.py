#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 16:10:13 2025

@author: danielhier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spaCy-Based Term Normalization Script
-------------------------------------

This script matches extracted terms to Human Phenotype Ontology (HPO) terms using vector similarity
via spaCy's `en_core_web_lg` model.

Overview of Functionality:
1. Load spaCy Model:
   - Loads the large English model `en_core_web_lg` for semantic vector representations.
2. Load Input CSV Files:
   - `hp_terms.csv`: must contain columns `hp_term` and `hp_id`.
   - `well_formed_terms.csv`: must contain a column `extracted_term`.
3. Precompute Vectors:
   - Computes and caches vector representations for all HPO terms to optimize performance.
4. Compute Similarity:
   - Uses cosine similarity between spaCy vectors to assess semantic closeness.
5. Find Closest Matches:
   - For each term in `well_formed_terms`, finds the best-matching `hp_term` by similarity.
6. Save Results:
   - Outputs a file `normalization_by_Spacy.csv` containing:
     - `well_formed_term`, `best_match`, `best_match_id`, `similarity`
7. Monitor Progress:
   - Uses `tqdm` progress bars for visibility during long-running computations.

Requirements:
- Python packages: spacy, pandas, numpy, tqdm
- spaCy model: Install with `python -m spacy download en_core_web_lg`

Notes:
- Cosine similarity ranges from -1 (opposite) to 1 (identical).
- Ensure CSV files are UTF-8 encoded and contain no missing values in required columns.
"""

import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

# Set paths relative to the scripts/ directory
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / 'data'
results_dir = base_dir / 'results'

# Input and output file paths
hp_terms_path = data_dir / 'hp_terms.csv'
well_formed_terms_path = data_dir / 'well_formed_terms.csv'
output_file_path = results_dir / 'normalization_by_Spacy.csv'

# Read the CSV files into Pandas dataframes
df_hp_terms = pd.read_csv(hp_terms_path)
df_well_formed_terms = pd.read_csv(well_formed_terms_path)

# Precompute vectors for HPO terms and save in a dictionary
hp_vectors = {}
for index, row in tqdm(df_hp_terms.iterrows(), total=df_hp_terms.shape[0], desc="Precomputing HPO term vectors"):
    term = row['hp_term']  # assuming 'hp_term' is the column name
    doc = nlp(term)
    hp_vectors[term] = doc.vector

# Function to compute similarity between two vectors using cosine similarity
def compute_similarity_vector(vector1, vector2):
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vector1, vector2) / (norm1 * norm2)

# Iterate through each row in well_formed_terms and find the closest match in hp_terms
closest_matches = []

for index, row in tqdm(df_well_formed_terms.iterrows(), total=df_well_formed_terms.shape[0], desc="Processing well-formed terms"):
    well_formed_term = row['extracted_term']
    well_formed_doc = nlp(well_formed_term)
    well_formed_vector = well_formed_doc.vector

    best_match = None
    best_match_id = None
    highest_similarity = -1

    for hp_term, hp_vector in hp_vectors.items():
        similarity = compute_similarity_vector(well_formed_vector, hp_vector)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = hp_term
            best_match_id = df_hp_terms[df_hp_terms['hp_term'] == hp_term]['hp_id'].values[0]

    closest_matches.append({
        'well_formed_term': well_formed_term,
        'best_match': best_match,
        'best_match_id': best_match_id,
        'similarity': highest_similarity
    })

# Convert the closest matches to a DataFrame
df_closest_matches = pd.DataFrame(closest_matches)

# Save the results to CSV
df_closest_matches.to_csv(output_file_path, index=False)

# Display the first few results
print(df_closest_matches.head())
print(f"Closest matches saved to {output_file_path}")
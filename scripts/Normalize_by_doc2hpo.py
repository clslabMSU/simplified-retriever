#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: hpo_term_extraction_doc2hpo_sentence_wrapped.py
Description:
This script loads phenotype terms from data/well_formed_terms.csv,
wraps each term in a clinical sentence for better parsing, and sends it to
the Doc2HPO API at /parse/acdat. It extracts HPO concepts and IDs and
saves the results to results/doc2hpo_extracted_terms.csv.

Requirements:
- Python packages: requests, pandas, tqdm
- Internet access to call https://doc2hpo.wglab.org/parse/annotate
"""

import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time

# Set up file paths relative to script location
project_root = Path(__file__).resolve().parents[1]
input_file = project_root / "data" / "well_formed_terms.csv"
output_file = project_root / "results" / "doc2hpo_extracted_terms.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)
Count =500 #Set the nummber of terms you wish to normalize
# Load the input terms
df = pd.read_csv(input_file)
terms = df['extracted_term'].tolist()

# Doc2HPO API endpoint
url = "https://doc2hpo.wglab.org/parse/acdat"

# Store results
results = []


i=1
# Loop through each term
for i, term in enumerate(tqdm(terms, desc="Extracting HPO terms from Doc2HPO")):
    if i < Count:
# Wrap term in clinical sentence
        sentence = f"The patient was examined and found to have {term}."
        i+=1
        data = {
                "note": sentence,
                "negex": True
            }
        
        try:
                response = requests.post(url, json=data)
                response.raise_for_status()
                result = response.json()
                print(result)
        
                if "hmName2Id" in result and result["hmName2Id"]:
                    for entry in result["hmName2Id"]:
                        hpo_term = entry.get("hpoName", "")
                        hpo_id = entry.get("hpoId", "")
                        negated = entry.get("negated", False)
                        results.append({
                            "extracted_term": term,
                            "hpo_term": hpo_term,
                            "hpo_id": hpo_id,
                            "negated": negated
                        })
                else:
                    results.append({
                        "extracted_term": term,
                        "hpo_term": "None detected",
                        "hpo_id": "None",
                        "negated": False
                    })
        
        except Exception as e:
                results.append({
                    "extracted_term": term,
                    "hpo_term": "ERROR",
                    "hpo_id": str(e),
                    "negated": False
                })
        
        time.sleep(0.3)  # Be gentle with the public API

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")

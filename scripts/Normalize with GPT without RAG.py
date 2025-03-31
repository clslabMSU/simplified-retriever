
"""
Created on Sun Jul 14 07:49:35 2024

@author: danielhier
# Explanation:
# This program is designed to find the best matching term from the Human Phenotype Ontology (HPO) for each given medical term.
# It utilizes the GPT-3.5-turbo model to identify the best match. The steps are as follows:
# 1. Load a list of target terms from a CSV file.
# 2. For each target term, query GPT-3.5-turbo to find the best matching HPO term and its corresponding HPO_ID.
# 3. Store the results, which include the target term, the best matching HPO term, and the HPO_ID.
# 4. Save the results to a new CSV file.

# Function to interact with OpenAI API
"""

import openai
from openai import OpenAI
import pandas as pd
import os
import json
from tqdm import tqdm
import openai
# Set up the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
client= openai
import openai
import pandas as pd
import os
import json
from tqdm import tqdm

# Substitute any valid Open AI model name here
model_name = 'gpt-4o'
num_rows =10  #Change this to increased number of rows of data normalized
# Function to interact with OpenAI API
def find_best_match(term):
    prompt = (
        f"You are given a term to normalize to a concept from the Human Phenotype Ontology and its HPO_ID.\n"
        f"Term: {term}\n"
        "Pick the best one and return the best one in JSON format as follows:\n"
        "{ \"normalized_term\": \"term\", \"normalized_id\": \"HP:xxxxxxx\" }\n"
        "If none of the terms in the HPO seem appropriate, you may return a value of \"no appropriate match\" and the HPO_ID \"HP:0000000\" in JSON format."
    )
    
    response = client.chat.completions.create(
        model= model_name,
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



target_terms_file_path = '/Users/danielhier/Documents/GitHub/simplified-retriever/data/well_formed_terms.csv'

target_df = pd.read_csv(target_terms_file_path)
print(target_df.columns)

target_terms = target_df['extracted_term'].tolist()

# Limit the process to the first 100 terms
target_terms = target_terms[:num_rows]

# Find the best HPO term for each target term
results = []
for term in tqdm(target_terms, desc="Finding best matches"):
    best_hpo_term = find_best_match(term)
    results.append((term, best_hpo_term['normalized_term'], best_hpo_term['normalized_id']))

# Create a DataFrame for the results
results_df = pd.DataFrame(results, columns=['extracted_term', 'normalized_term', 'normalized_id'])

# Save the results to a new CSV file
output_path = f'/Users/danielhier/Desktop/HPO_V2/hpo_terms_with_normalizations_{model_name}.csv'
results_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
print(results_df.head())













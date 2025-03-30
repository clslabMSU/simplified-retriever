"""
Documentation for HPO Word Vector Generator

Overview

This script reads a CSV file containing Human Phenotype Ontology (HPO) terms and their corresponding IDs, generates word vectors for each term using BioBERT embeddings, and saves the results as both a Pickle and a JSON file. These word vectors can be used for machine learning, NLP tasks, or semantic analysis of HPO terms.

Installation Instructions

1. Prerequisites

Ensure you have Python installed (version 3.8 or higher is recommended). To check your Python version, run:

python --version

2. Required Python Libraries

The following libraries are required to run this program:
	•	pandas: For handling and manipulating tabular data from CSV files.
	•	transformers: To load the BioBERT model and tokenizer.
	•	torch: PyTorch is required to support the deep learning computations of the BioBERT model.
	•	tqdm: For displaying a progress bar when processing multiple HPO terms.

3. Installation Commands

To install all required libraries, you can run the following commands:

# Install pandas for data manipulation
pip install pandas

# Install PyTorch (CUDA version may vary depending on your system)
pip install torch

# Install HuggingFace Transformers to load the BioBERT model and tokenizer
pip install transformers

# Install tqdm for a progress bar
pip install tqdm

	Note: If you have a CUDA-compatible GPU, you can install the GPU-accelerated version of PyTorch. Visit https://pytorch.org/get-started/locally/ for instructions.

Input File

The script reads a CSV file (HP0.csv) containing HPO terms and their HPO IDs. The structure of the file should look like this:

HPO_ID	HPO_Term
HP:0001250	Seizure
HP:0001263	Tremor
HP:0001290	Spasticity
…	…

Required Columns
	•	HPO_ID: The unique identifier for the HPO term (e.g., HP:0001250).
	•	HPO_Term: The human-readable label for the HPO term (e.g., “Seizure”).

	Note: Ensure that the column names in your CSV file match these headers exactly. If the headers are different, you may need to modify the script accordingly.

How It Works
	1.	Data Loading: The script loads the CSV file containing the HPO terms and their HPO IDs using the pandas library.
	2.	BioBERT Embeddings: For each HPO term, it generates a word vector using BioBERT. The following key steps happen:
	•	The HPO term is tokenized using AutoTokenizer from HuggingFace Transformers.
	•	The tokenized term is fed into the BioBERT model to get the contextual embeddings.
	•	The average of the embeddings for all tokens in the term is taken as the word vector.
	3.	Data Export: The script exports the final DataFrame to two formats:
	•	Pickle file: Serialized Python object for fast loading and saving.
	•	JSON file: Human-readable JSON format.

File Paths

You can customize the file paths in the script to suit your system.

Input File

file_path = '/Users/danielhier/Desktop/HPO/HP0.csv'

	•	Replace this path with the location of your HPO CSV file.

Output Files

The following two files are generated:
	1.	Pickle File (preferred for large, complex data):

output_pickle_path = '/Users/danielhier/Desktop/HPO/HPO_with_word_vectors.pkl'

This file can be loaded quickly using pandas.read_pickle().

	2.	JSON File (optional, for interoperability):

output_json_path = '/Users/danielhier/Desktop/HPO/HPO_with_word_vectors.json'

This file can be viewed and shared across different platforms.

Code Walkthrough

# Import necessary libraries
import pandas as pd  # Used to read/write CSV, JSON, and Pickle files
from transformers import AutoTokenizer, AutoModel  # Load BioBERT tokenizer and model
import torch  # Required for tensor-based operations
from tqdm import tqdm  # Provides a progress bar

1. Load the CSV File

# Load the CSV file
file_path = '/Users/danielhier/Desktop/HPO/HP0.csv'
df = pd.read_csv(file_path)

This reads the CSV file containing the HPO terms and HPO IDs into a pandas DataFrame df.

2. Load BioBERT Tokenizer and Model

# Load BioBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

This loads the BioBERT tokenizer and model. The model used is dmis-lab/biobert-base-cased-v1.1, a popular choice for biomedical text processing.

3. Generate Word Vectors

def get_word_vector(term):
    inputs = tokenizer(term, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    word_vector = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return word_vector

For each HPO term, this function:
	1.	Tokenizes the text.
	2.	Passes it through BioBERT.
	3.	Averages the token embeddings to produce a single word vector for the term.

4. Process Each Term

word_vectors = []
for term in tqdm(df['HPO_Term'], desc="Processing HPO Terms"):
    word_vectors.append(get_word_vector(term))

	•	A progress bar (tqdm) shows the progress of processing HPO terms.
	•	For each term in the HPO_Term column, it generates the word vector and appends it to the list.

5. Export Data

# Create a DataFrame with word vectors
word_vectors_df = pd.DataFrame({
    "HPO_ID": df['HPO_ID'],
    "HPO_Term": df['HPO_Term'],
    "Word_Vector": word_vectors
})

A new DataFrame is created with the original HPO information plus the word vectors.

# Save as Pickle
word_vectors_df.to_pickle(output_pickle_path)

# Save as JSON
word_vectors_df.to_json(output_json_path, orient='records')

print(f"Saved Pickle to {output_pickle_path}")
print(f"Saved JSON to {output_json_path}")

The data is saved as both a Pickle file and a JSON file.

Usage

To run the script, execute the following command in your Python environment:

python your_script_name.py

	Tip: If you want to speed up the process, consider using CUDA (if you have an NVIDIA GPU) by installing PyTorch with CUDA support.

Potential Enhancements
	1.	Speed Up Embedding Calculation: Use GPU acceleration with PyTorch (requires CUDA-enabled hardware).
	2.	Batch Processing: Instead of one term at a time, process batches of terms to reduce the overhead of calling the model multiple times.
	3.	Error Handling: Add error handling to skip any terms that cause issues with tokenization.
	4.	Data Integrity Check: Ensure the input CSV file has the required HPO_ID and HPO_Term columns.

Example Output

Sample Pickle Data

HPO_ID      HPO_Term        Word_Vector
HP:0001250  Seizure         [0.123, 0.456, 0.789, ...]
HP:0001263  Tremor          [0.234, 0.567, 0.890, ...]
HP:0001290  Spasticity      [0.345, 0.678, 0.901, ...]

Support & Troubleshooting

If you encounter issues such as out of memory errors or model loading issues, try the following:
	•	Batch Process Terms to avoid memory overload.
	•	Use CPU Only if GPU memory is insufficient (CUDA_VISIBLE_DEVICES="").
	•	Ensure you have correct permissions for the file paths used for input and output.

This documentation provides an end-to-end guide for setting up, running, and extending the script for generating word vectors from HPO terms using BioBERT. Let me know if you’d like any adjustments or explanations! 😊

"""

import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# Load the CSV file
file_path = '/Users/danielhier/Desktop/HPO/HP0.csv'
df = pd.read_csv(file_path)

# Load BioBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Function to calculate the word vector for a given term
def get_word_vector(term):
    inputs = tokenizer(term, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    word_vector = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return word_vector

# Calculate word vectors for each term with a progress bar
word_vectors = []
for term in tqdm(df['HPO_Term'], desc="Processing HPO Terms"):
    word_vectors.append(get_word_vector(term))

# Create a new DataFrame with word vectors
word_vectors_df = pd.DataFrame({
    "HPO_ID": df['HPO_ID'],
    "HPO_Term": df['HPO_Term'],
    "Word_Vector": word_vectors
})

# Save as Pickle (preferred for complex data)
output_pickle_path = '/Users/danielhier/Desktop/HPO/HPO_with_word_vectors.pkl'
word_vectors_df.to_pickle(output_pickle_path)

# Optional: Save as JSON
output_json_path = '/Users/danielhier/Desktop/HPO/HPO_with_word_vectors.json'
word_vectors_df.to_json(output_json_path, orient='records')

print(f"Saved Pickle to {output_pickle_path}")
print(f"Saved JSON to {output_json_path}")
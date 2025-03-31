##**simplified-retriever**

Large language models (LLMs) have shown improved accuracy in phenotype term normalization tasks when augmented with retrievers that suggest candidate matches based on semantic similarity.

In this project, we introduce a simplified retriever that enhances LLM performance by searching the Human Phenotype Ontology (HPO) for candidate matches using contextual word embeddings from BioBERT — without requiring explicit term definitions.

When tested on phenotype terms extracted from the clinical synopses in *Online Mendelian Inheritance in Man (OMIM®)*, we observed that normalization accuracy with GPT-4o increased from a baseline of **62%** (no augmentation) to **85%** with retriever augmentation.

This approach is generalizable to other biomedical term normalization tasks and offers an efficient alternative to more complex retrieval pipelines.

---

## 📦 Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Download the required spaCy model (if using the spaCy-based script):

```bash
python -m spacy download en_core_web_lg
```

If using the GPT-based normalization script, set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

---

## 📁 Repository Structure

```
data/       # Input term lists and precomputed BioBERT similarity candidates
results/    # Output files from spaCy and GPT-4o normalization scripts
scripts/    # Python scripts for retriever + normalization pipeline
```

---

## 📂 Files and Scripts

### `data/well_formed_terms.csv`
- CSV file with the header `extracted_term`
- Contains 1,820 phenotype terms extracted from OMIM summaries

### `scripts/Generate_BioBERT_word_vectors_for_HPO_terms.py`
- Generates a BioBERT embedding for each term in the HPO

### `data/HPO.csv`
- 18,988 rows of HPO term metadata  
- Column 1: `hp_id`, Column 2: `hp_term`

### `scripts/Normalize_by_Spacy.py`
- Normalizes terms using cosine similarity from **spaCy**'s `en_core_web_lg` vector model

### `scripts/Normalize_with_GPT_RAG.py`
- Normalizes terms using **GPT-4o** with Retrieval-Augmented Generation (RAG)  
- Requires OpenAI API key  
- Uses precomputed BioBERT similarity scores for candidate terms


### `Plot_model_accuracies.py`
- Plots a bar chart of model accuracies

### `scripts/Plot_rag_accuracy_vs_candidates.py`
- Plots a bar chart of RAG model accuracies with number of candidates as a indeependent variable on x-axis
---`


## ✅ Outputs

Example output files:

- `results/normalization_by_Spacy.csv`
- `results/best_matches_gpt_4o_with_RAG_20_candidates_1.csv`

Each output contains:
- `target_term`: input term from OMIM  
- `best_hpo_term`: top match selected (via spaCy or GPT-4o)  
- `hpo_id`: matched HPO concept ID

---

## 🧠 Citation

This repository supports the findings of the published manuscript:

Hier DB, Do TS, Obafemi-Ajayi T. 
A simplified retriever to improve accuracy of phenotype normalizations by large language models. 
Frontiers in Digital Health. 2025 Mar 4;7:1495040.
https://doi.org/10.3389/fdgth.2025.1495040

# simplified-retriever
Large language models have shown improved accuracy in phenotype term normalization tasks when augmented with retrievers that suggest candidate normalizations based on term definitions. In this work, we introduce a simplified retriever that enhances large language model accuracy by searching the Human Phenotype Ontology (HPO) for candidate matches using contextual word embeddings from BioBERT without the need for explicit term definitions. Testing this method on terms derived from the clinical synopses of Online Mendelian Inheritance in Man (OMIM®), we demonstrate that the normalization accuracy of GPT-4o increases from a baseline of 62% without augmentation to 85% with retriever augmentation. This approach is potentially generalizable to other biomedical term normalization tasks and offers an efficient alternative to more complex retrieval methods.

Files:
1) **well_formed_terms.csv** is a csv file with the header 'extracted_term'. It has 1820 rows (terms) all found in OMIM summaries and all are candidates to be normalized by being mapped to an HPO concept. This is our primary test file of terms.

2) **Generate_ BIOBERT_word_vectors_for_HPO_terms.py** is a python program that generates a BioBERT embedding for each term in the HPO.

3) **HPO.csv** is a CSV file with 18,988 HPO_Terms and their HPO_ID.  Column 1 is HPO_ID. Column 2 is HPO_Term 

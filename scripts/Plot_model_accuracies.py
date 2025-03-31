#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 08:18:57 2024

@author: danielhier

Generates a horizontal bar chart comparing the accuracy of various normalization models.
Saves output to results/model_accuracies.png
"""

import matplotlib.pyplot as plt
from pathlib import Path

# Define relative path to results folder
output_dir = Path(__file__).resolve().parents[1] / "results"
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
output_file = output_dir / "model_accuracies.png"

# Data
models = [
    'GPT-4o-mini', 'GPT-3.5-turbo', 'GPT-4o', 'spaCy',
    'BioBERT', 'GPT-4o-mini with RAG', 'GPT-3.5-turbo with RAG', 'GPT-4o with RAG'
]
accuracies = [13.2, 50.9, 62.3, 50.3, 70.3, 70.6, 89.5, 88.3]

# Create the horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(models, accuracies, color='blue')
plt.xlabel('Accuracy (%)')
plt.title('Model Accuracies')
plt.xlim(0, 100)
plt.gca().invert_yaxis()

# Save the figure
plt.savefig(output_file, bbox_inches='tight', dpi=600)

# Show the plot
plt.show()
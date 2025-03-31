#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 07:06:21 2024

@author: danielhier

This script plots the improvement in normalization accuracy of GPT-4o with increasing
number of RAG (retrieval-augmented generation) candidates included in the prompt.

Saves output to: results/influence_of_rag_choices_on_accuracy.png
"""

import matplotlib.pyplot as plt
from pathlib import Path

# Create results directory if it doesn't exist
output_dir = Path(__file__).resolve().parents[1] / "results"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "influence_of_rag_choices_on_accuracy.png"

# Data
choices = [1, 2, 5, 10, 20, 30, 40, 50]
accuracy = [68.6, 75.9, 84.1, 84.5, 87.5, 87.5, 88.0, 88.0]

# Create the line chart
plt.figure(figsize=(10, 6))
plt.plot(choices, accuracy, marker='o')

# Add labels and title
plt.title('Effect of RAG Candidate Count on Accuracy')
plt.xlabel('Number of RAG Candidates in Prompt')
plt.ylabel('Accuracy (%)')

# Add grid for better readability
plt.grid(True)

# Save the figure
plt.savefig(output_path, bbox_inches='tight', dpi=600)

# Show the plot
plt.show()

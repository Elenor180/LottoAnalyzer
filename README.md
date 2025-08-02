# Lotto Combo Analyzer

🎯 A comprehensive Python & Streamlit app for analyzing and predicting lottery number combinations using clustering, association rule mining, and LSTM deep learning.

---

## Overview

Lotto Combo Analyzer processes historical lottery data to uncover patterns and predict likely next number combinations. The app combines statistical analysis, machine learning, and interactive visualizations in a user-friendly dashboard.

---

## Features

- Load and preprocess historical lotto data robustly.
- Validate uniqueness of newly generated combinations.
- Analyze most frequent numbers and number combinations.
- Cluster historical data with K-Means for pattern discovery.
- Discover association rules between numbers using Apriori algorithm.
- Predict next number combos using LSTM neural networks.
- Generate multiple unique and plausible combinations.
- Interactive, styled interface built with Streamlit.

---


## Architecture and Machine Learning Integration

The Lotto Combo Analyzer is designed with a modular architecture that separates data processing, analysis, machine learning, and user interface components for clarity and maintainability.

- **Data Layer:** Loads and preprocesses historical lottery data from CSV files, detecting relevant numeric columns and converting each lotto draw into sorted tuples for consistency.
  
- **Analysis Layer:** Performs statistical analyses such as frequency counts of numbers and combinations, clustering with K-Means to identify natural groupings, and association rule mining using the Apriori algorithm to uncover relationships between numbers.

- **Machine Learning Layer:** Utilizes a Long Short-Term Memory (LSTM) neural network model built with TensorFlow/Keras to predict the next likely lottery number based on historical frequency sequences. The LSTM captures temporal dependencies and patterns in the data that simple statistics might miss.

- **Combination Generation:** The model's predictions are combined with a frequency-based heuristic to generate unique, plausible lottery combinations that have not appeared historically.

- **Presentation Layer:** A Streamlit web interface styled with custom CSS provides an interactive dashboard to view data, validate combinations, analyze patterns, and display predictions, ensuring a seamless user experience.

This architecture balances classic data mining techniques with modern deep learning to provide robust and insightful lottery number analysis and predictions.


## Components & Techniques

- **Data Processing:** Loading CSVs, dynamic column detection, sorting combos.
- **Uniqueness Check:** Efficient set lookups for duplicate detection.
- **Frequency Analysis:** Counting occurrences with `Counter` and visualizing with seaborn.
- **Clustering:** K-Means clustering and cluster visualization.
- **Association Rules:** Mining frequent itemsets and rules with `mlxtend`.
- **Deep Learning:** Sequence modeling and prediction via TensorFlow/Keras LSTM.
- **Combination Generation:** Blending ML predictions with frequency-ranked numbers.
- **UI/UX:** Streamlit with custom CSS for polished look and ease of use.

---

## Challenges & Solutions

- Handling missing or inconsistent data robustly.
- Generating unique combinations respecting constraints.
- Merging ML predictions with frequency heuristics effectively.
- Maintaining app responsiveness with caching and efficient operations.
- Designing intuitive UI with clear feedback and styling.

---

## Skills Developed

- Data science, machine learning, and deep learning.
- Python libraries: pandas, numpy, scikit-learn, mlxtend, tensorflow, seaborn, matplotlib.
- Software engineering: modular code, error handling, caching.
- UI design using Streamlit and CSS.
- Problem solving in data validation, ML integration, and UX.

---

## Getting Started

1. Clone this repository.
2. Install dependencies with:

   ```bash
   pip install -r requirements.txt

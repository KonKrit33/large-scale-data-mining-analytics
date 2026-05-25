# Large-Scale Data Mining & Similarity Search

This repository contains a project developed for the High Scale Analytics course of the MSc in Data Science & Information Technologies. The project focuses on large-scale data mining, machine learning, similarity search, and time-series similarity analysis using Python.

## Project Overview

The project includes three main tasks:

1. **Text Classification**
   - News article classification into four categories: Business, Entertainment, Health, and Technology.
   - Feature extraction using Bag of Words.
   - Model evaluation using Support Vector Machines, KNN with Jaccard similarity, and 5-fold cross-validation.

2. **Nearest Neighbor Search with MinHash LSH**
   - Implementation of nearest-neighbor search using Jaccard similarity.
   - Comparison between brute-force search and MinHash Locality Sensitive Hashing.
   - Evaluation based on build time, query time, total time, and retrieval accuracy.

3. **Dynamic Time Warping**
   - Custom implementation of Dynamic Time Warping for measuring similarity between time series with different resolutions.
   - Output generation for DTW distances.

## Repository Structure

```text
.
├── Project_Q1_1.py
├── Project_Q1_2_a.py
├── Project_Q1_2_b.py
├── Project_Q1_2_c.py
├── Project_Q2_a.py
├── Project_Q2_b.py
├── Project_Q3.py
├── testSet_categories.csv
├── dtw.csv
└── Report.pdf
```

## Technologies Used

- Python
- scikit-learn
- Keras
- NumPy
- Pandas
- Bag of Words
- Support Vector Machines
- KNN
- Jaccard Similarity
- MinHash
- Locality Sensitive Hashing
- Dynamic Time Warping

## Notes

The full input datasets are not included in this repository due to size and distribution restrictions. The repository includes the source code, output files, and final report documenting the methodology, experiments, mathematical background, and results.

## Keywords

Machine Learning, Data Mining, Text Classification, Similarity Search, Locality Sensitive Hashing, MinHash, Dynamic Time Warping, Time-Series Similarity, Python, High-Scale Analytics.

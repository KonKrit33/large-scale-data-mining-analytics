# Large-Scale Text Classification, Similarity Search and Time-Series Alignment

This repository contains a Python-based data mining and machine learning portfolio project focused on large-scale text classification, approximate similarity search, and time-series alignment.

The project combines classical machine learning, text preprocessing, nearest-neighbor search, Locality Sensitive Hashing, and a custom Dynamic Time Warping implementation.

## Project Overview

The repository includes three independent but related workflows:

1. **News Text Classification**

   * Classification of news articles into four categories: Business, Entertainment, Health, and Technology.
   * Text preprocessing using lowercasing, punctuation removal, stopword filtering, and token cleaning.
   * Feature extraction using Bag-of-Words and TF-IDF representations.
   * Model evaluation using Linear Support Vector Machines, KNN with Jaccard similarity, and stratified 5-fold cross-validation.
   * Final prediction generation using the best-performing TF-IDF + Linear SVM model.

2. **Scalable Similarity Search with MinHash LSH**

   * Implementation of nearest-neighbor search using Jaccard similarity over binary Bag-of-Words vectors.
   * Comparison between exact brute-force search and approximate MinHash Locality Sensitive Hashing.
   * Evaluation based on index build time, query time, total runtime, and nearest-neighbor recall.
   * Analysis of how LSH parameters affect the trade-off between speed and retrieval quality.

3. **Custom Dynamic Time Warping**

   * Custom implementation of Dynamic Time Warping for one-dimensional time-series alignment.
   * Dynamic programming implementation without external DTW libraries.
   * Support for time series of different lengths.
   * Memory-efficient computation using two-row dynamic programming.
   * Output generation for pairwise DTW distances.

## Repository Structure

```text
.
├── news_category_wordcloud_analysis.py
├── news_category_classification_svm_knn.py
├── optimized_news_classification_tfidf_svm.py
├── predict_news_categories_tfidf_svm.py
├── news_similarity_search_lsh_threshold.py
├── news_similarity_search_lsh_fixed_rows.py
├── custom_dynamic_time_warping.py
├── news_category_predictions.csv
├── dtw_distances.csv
├── Large_Scale_Data_Mining_Report.pdf
├── README.md
└── .gitignore
```

## Main Components

### News Category Word Cloud Analysis

This script performs exploratory text analysis by generating category-specific word clouds for news articles.

It includes:

* text cleaning,
* stopword removal,
* category-wise corpus construction,
* word frequency visualization,
* generation of word cloud images for each news category.

### News Category Classification

This workflow compares multiple machine learning approaches for text classification.

It includes:

* Bag-of-Words feature extraction,
* Linear SVM classification,
* KNN classification using Jaccard similarity,
* stratified 5-fold cross-validation,
* accuracy-based model comparison.

### Optimized TF-IDF Classification Model

This script implements an improved text classification pipeline using TF-IDF features and a tuned Linear SVM classifier.

It includes:

* unigram and bigram TF-IDF features,
* sublinear term-frequency scaling,
* optimized Linear SVM configuration,
* comparison against baseline models.

### MinHash LSH Similarity Search

This workflow compares exact and approximate nearest-neighbor search over text documents.

It includes:

* binary Bag-of-Words representation,
* exact brute-force Jaccard search,
* MinHash signatures,
* Locality Sensitive Hashing,
* retrieval recall estimation,
* runtime comparison across LSH configurations.

### Dynamic Time Warping

This script implements Dynamic Time Warping from scratch for time-series similarity analysis.

It includes:

* safe parsing of time-series inputs,
* Euclidean local cost computation,
* dynamic programming recurrence,
* memory-efficient two-row implementation,
* CSV output generation for DTW distances.

## Tools and Technologies

* Python
* Pandas
* NumPy
* scikit-learn
* SciPy
* datasketch
* WordCloud
* Matplotlib
* Bag-of-Words
* TF-IDF
* Linear Support Vector Machines
* KNN
* Jaccard Similarity
* MinHash
* Locality Sensitive Hashing
* Dynamic Time Warping

## Technical Relevance

This repository demonstrates practical skills in:

* large-scale text preprocessing,
* machine learning model evaluation,
* feature engineering for text data,
* similarity search over high-dimensional sparse representations,
* approximate nearest-neighbor retrieval,
* time-series alignment,
* algorithmic implementation,
* reproducible Python-based data analysis.

The workflows are relevant to:

* machine learning engineering,
* data mining,
* information retrieval,
* scalable analytics,
* natural language processing,
* time-series similarity analysis,
* industrial data analytics,
* AI-ready data preprocessing pipelines.

## Notes

The full input datasets are not included due to size and distribution restrictions.

The repository contains source code, selected output files, and a technical report documenting the methodology, experiments, mathematical background, and results.

All scripts use relative paths so that the repository can be cloned and executed locally after placing the required input files in the project directory.

## Keywords

Machine Learning, Data Mining, Text Classification, Natural Language Processing, Similarity Search, Information Retrieval, Jaccard Similarity, MinHash, Locality Sensitive Hashing, Dynamic Time Warping, Time-Series Similarity, Python, Scalable Analytics.

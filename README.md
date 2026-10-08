# Hotel Review Sentiment Analysis

## Project Overview

This project applies **Natural Language Processing (NLP)** and
**traditional Machine Learning** to hotel reviews. The system predicts
overall hotel ratings (1--5 stars), sentiment polarity
(Negative/Neutral/Positive), and ratings for available hotel aspects.

> **Note:** This is a paper-style reproduction/approximation using a
> smaller public substitute dataset. It is not an exact reproduction of
> the original study.

## Problem Statement

Hotel reviews contain valuable information about customer satisfaction,
but manually analyzing large numbers of reviews is time-consuming. This
project investigates whether machine-learning models can automatically
learn from review text to predict numerical ratings, sentiment, and
aspect-level ratings.

## Dataset

The project uses:

`tripadvisor_sample_1000.csv`

It contains **1,000 English hotel reviews** with these fields:

  Column            Meaning
  ----------------- --------------------------
  `title`           Review title
  `text`            Review body
  `overall`         Overall rating from 1--5
  `rooms`           Room rating
  `location`        Location rating
  `value`           Value-for-money rating
  `cleanliness`     Cleanliness rating
  `sleep_quality`   Sleep-quality rating

The title and review body are combined into `review_text`.

### Preprocessing

-   Convert ratings to numeric values.
-   Keep valid ratings from 1 to 5.
-   Remove empty review text.
-   Combine title + body.
-   Use only available labels for each aspect.
-   Split data into approximately **2/3 training and 1/3 testing**.
-   Use `random_state=42` for reproducibility.

## Methodology

``` text
Raw Reviews
    ↓
Data Cleaning
    ↓
Title + Review Body
    ↓
Train/Test Split
    ↓
Text Vectorization
    ↓
Machine Learning Model
    ↓
Rating / Sentiment Prediction
    ↓
Evaluation
```

### Feature Representations

The project compares:

1.  **Word unigrams** --- individual words.
2.  **Word 2/3-grams** --- two- and three-word phrases such as
    `not worth` or `very clean room`.
3.  **Character 5-grams** --- sequences of five characters that can
    capture parts of words and spelling patterns.

Both **raw counts** and **TF-IDF** representations are tested.

### Algorithms

-   **Multinomial Naive Bayes**
-   **Linear Support Vector Machine (Linear SVM)**
-   **Linear Regression**

The Linear SVM uses:

``` text
C = 1
max_iter = 5000
```

### Sentiment Mapping

  Rating   Polarity
  -------- ----------
  1--2     Negative
  3        Neutral
  4--5     Positive

## Class Balancing

Four strategies were compared using word unigram + TF-IDF + Linear SVM:

  Strategy                Star Accuracy
  --------------------- ---------------
  No balance                      55.4%
  **Class weighting**         **57.2%**
  Oversampling                    55.7%
  Undersampling                   47.9%

Class weighting performed best.

## Feature Experiment Results

For the overall rating target:

  Feature                Star Accuracy   Polarity Accuracy
  -------------------- --------------- -------------------
  1-word counts                  54.8%               80.5%
  1-word TF-IDF              **57.2%**               80.5%
  5-character counts             53.9%               79.3%
  5-character TF-IDF         **57.2%**           **81.1%**
  2/3-word counts                53.9%               79.3%
  2/3-word TF-IDF                54.5%               79.0%

The best polarity result was **81.1%** with 5-character TF-IDF. The
highest exact star accuracy was **57.2%**, achieved by both unigram
TF-IDF and character 5-gram TF-IDF.

## Algorithm Comparison

Using word unigram + TF-IDF:

  Algorithm                   Star Accuracy   Polarity Accuracy
  ------------------------- --------------- -------------------
  Multinomial Naive Bayes             48.8%               78.7%
  **Linear SVM**                  **57.2%**               80.5%
  Linear Regression                   53.6%               80.8%

Linear SVM was the strongest algorithm for exact 1--5 star prediction in
this comparison.

## Final Paper-Style Model

The final paper-style pipeline is:

``` text
Word 2/3-Gram Count Features
        ↓
Class-Weighted Linear SVM
        ↓
Rating / Sentiment Prediction
```

Its balancing-experiment result was:

-   **57.2% star accuracy**
-   **80.5% polarity accuracy**

Do not confuse this with the separate feature experiment where character
5-gram TF-IDF achieved 81.1% polarity accuracy.

## Aspect-Level Analysis

Available aspects include:

-   Rooms
-   Location
-   Value
-   Cleanliness
-   Sleep Quality

Selected aspect results:

  Aspect            Overall-trained   Aspect-trained
  --------------- ----------------- ----------------
  Rooms                       53.6%            47.3%
  Location                    56.3%        **63.8%**
  Value                   **46.1%**            42.8%
  Cleanliness                 59.3%        **61.1%**
  Sleep Quality           **54.2%**            52.1%

## Evaluation Metrics

The project uses:

-   **Accuracy** --- percentage of correct predictions.
-   **Balanced Accuracy** --- gives more equal importance to different
    classes.
-   **Macro F1** --- averages F1 across classes equally.

## Technologies

-   Python
-   Jupyter Notebook
-   Pandas
-   NumPy
-   Matplotlib
-   Scikit-learn
-   Git
-   GitHub

Important scikit-learn components include:

`CountVectorizer`, `TfidfTransformer`, `LinearSVC`, `MultinomialNB`,
`LinearRegression`, `Pipeline`, `train_test_split`, `accuracy_score`,
`balanced_accuracy_score`, and `f1_score`.

## Installation and Execution

### 1. Clone the repository

``` bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

### 2. Create a virtual environment

Windows:

``` bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

If `requirements.txt` is present:

``` bash
pip install -r requirements.txt
```

Otherwise:

``` bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### 4. Start Jupyter

``` bash
jupyter notebook
```

Open:

``` text
hotel_review_sentiment_reproduction.ipynb
```

### 5. Dataset

Place:

``` text
tripadvisor_sample_1000.csv
```

where the notebook expects it.

### 6. Run

Run the notebook cells from top to bottom. It performs data cleaning,
feature extraction, model training, balancing experiments, algorithm
comparison, aspect analysis, and evaluation.

## Recommended Repository Structure

``` text
hotel-review-sentiment-analysis/
├── hotel_review_sentiment_reproduction.ipynb
├── tripadvisor_sample_1000.csv
├── requirements.txt
├── README.md
└── results/
    └── figures/
```

If the dataset cannot be shared, do not upload it. Instead, provide
instructions for obtaining/placing it locally.

## Example Prediction

The demonstration review `Convenient stay` produced:

  Target            Rating Polarity
  --------------- -------- ----------
  Overall                4 Positive
  Rooms                  5 Positive
  Location               5 Positive
  Value                  3 Neutral
  Cleanliness            5 Positive
  Sleep Quality          4 Positive

## Limitations

1.  The working dataset contains only 1,000 reviews.
2.  The original reference study used a much larger dataset.
3.  The substitute dataset does not contain every original aspect.
4.  Results may differ because of dataset size and composition.
5.  A random review-level split can place reviews from the same hotel in
    both partitions.
6.  Aspect models can produce predictions even when an aspect is not
    explicitly discussed.
7.  The project uses traditional ML rather than transformer models such
    as BERT.

## Conclusion

The project demonstrates a complete traditional NLP pipeline for
hotel-review analysis. It compares word, phrase, and character-based
representations and evaluates multiple machine-learning algorithms and
class-balancing methods.

Linear SVM provided the strongest exact star-rating performance among
the tested algorithms. Class weighting was the best tested balancing
strategy. Character 5-gram TF-IDF achieved the highest polarity accuracy
in the feature comparison.

## Course / Submission

**Course:** UE24CS352A -- Machine Learning\
**Institution:** PES University

Before submission, add:

-   Team member names
-   USNs
-   GitHub repository link
-   Individual contributions

The course guidelines require the source code to be maintained in a
private GitHub repository with a README containing setup/run
instructions.

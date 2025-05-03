# Text Emotion Classification Project

## Project Description

This project aims to classify text into distinct emotional categories. Utilizing machine learning techniques, the system processes raw text data, extracts relevant features, and trains various classification models to predict the underlying emotion expressed in the text. The goal is to build an accurate and robust model for automated emotion detection.

## Problem Solved

The ability to automatically identify emotions in text has numerous applications, including sentiment analysis, understanding customer feedback, analyzing social media trends, and improving human-computer interaction. This project addresses the challenge of accurately classifying text into a predefined set of emotion categories.

## Dataset

The project uses the "Emotions Dataset for NLP," likely sourced from a platform like Kaggle. The dataset is divided into three files:
- `train.txt` (16,000 entries)
- `val.txt` (2,000 entries)
- `test.txt` (2,000 entries)

Each entry consists of a text string and its corresponding emotion label, separated by a semicolon.

**Note:** Based on the Exploratory Data Analysis (EDA) showing low representation of 'love' and 'surprise' emotions, these categories were removed from all datasets to potentially improve model performance and focus on the more represented classes (joy, sadness, anger, fear).

## Methodology

The project follows a standard machine learning pipeline:

1.  **Data Loading & Initial Inspection:** Load the datasets and perform basic checks on their structure and content (e.g., data types, non-null counts).
2.  **Exploratory Data Analysis (EDA):** Analyze emotion class distribution using value counts and percentages to identify imbalances. Determine characteristics like maximum text length.
3.  **Data Cleaning:** Remove data points corresponding to underrepresented emotion classes ('love', 'surprise') to create a dataset focused on the primary emotions (joy, sadness, anger, fear).
4.  **Text Preprocessing:** Implement a custom `TextProcessor` using NLTK to:
    *   Remove non-alphabetic characters.
    *   Convert text to lowercase.
    *   Tokenize the text.
    *   Remove standard English stopwords.
    *   (Optional) Apply Porter Stemming (used for the Multinomial Naive Bayes model).
5.  **Feature Extraction:** Convert the processed text into numerical features using `CountVectorizer`, limiting the vocabulary to the top 3000 most frequent words.
6.  **Model Training & Evaluation:**
    *   Train and evaluate several traditional machine learning classification models (Random Forest, Support Vector Machine, Logistic Regression, Multinomial Naive Bayes, Gradient Boosting) using scikit-learn Pipelines.
    *   Evaluate each model's performance using Accuracy on train, validation, and test sets.
    *   Visualize classification performance using Confusion Matrices for train, validation, and test sets.
7.  **Model Comparison:** Compare the test set accuracies of all individual models using a bar plot.
8.  **Ensemble Modeling:** Build a `VotingClassifier` (using a 'hard' voting strategy) by combining the predictions of the best-performing individual models (Random Forest, Logistic Regression, SVM).
9.  **Final Validation:** Evaluate the ensemble model's final accuracy and test its ability to predict emotions on new, custom text inputs.

## Key Technologies and Libraries

*   Python
*   NumPy
*   Pandas
*   Matplotlib
*   Seaborn
*   NLTK (Natural Language Toolkit)
*   Scikit-learn (BaseEstimator, TransformerMixin, Pipelines, Vectorizers, various classification models, VotingClassifier)

## Setup and Installation

1.  **Clone the repository** (if applicable) or download the project files.
2.  **Ensure Python is installed** on your system (Python 3.6+ recommended).
3.  **Install the required libraries** using pip:
    ```bash
    pip install numpy pandas matplotlib seaborn nltk scikit-learn
    ```
4.  **Download NLTK data:** Run a Python interpreter or script and execute the following to download the necessary NLTK data (stopwords):
    ```python
    import nltk
    nltk.download('stopwords')
    # If using stemming, you might also need 'punkt'
    # nltk.download('punkt')
    ```

## How to Run the Project

1.  **Obtain the dataset files:** Download the `train.txt`, `val.txt`, and `test.txt` files from the "Emotions Dataset for NLP" (e.g., on Kaggle) and place them in a directory accessible by your script. **Note:** The paths in the code (`/kaggle/input/emotions-dataset-for-nlp/`) need to be updated to the actual paths on your system or environment.
2.  **Save the provided Python code:** Save the entire Python code block as a single `.py` file (e.g., `emotion_classifier.py`) or run it within a Jupyter Notebook environment.
3.  **Update file paths:** Modify the `file_path` variables at the beginning of the data loading section in the Python code to point to the correct locations of your `train.txt`, `val.txt`, and `test.txt` files.
4.  **Execute the script/notebook:** Run the Python script from your terminal or execute the cells in your Jupyter Notebook.

The script will perform the data loading, preprocessing, model training, evaluation, comparison, and final ensemble modeling steps sequentially.

## Results

The project evaluated the performance of several models on the test set after removing 'love' and 'surprise' emotions.

*   The **Logistic Regression** model achieved the highest accuracy among individual classifiers (~93.92%).
*   The implemented **Voting Classifier** ensemble achieved the highest overall test accuracy of **~94.31%**.

This demonstrates the effectiveness of combining multiple models through ensemble learning for this classification task. The confusion matrices provide detailed insights into which emotion categories were most accurately predicted and where common misclassifications occurred. The final validation on custom text shows the model's practical application.

## Future Enhancements

*   Experiment with different text vectorization techniques (e.g., TF-IDF, Word Embeddings like Word2Vec, GloVe, or contextual embeddings from models like BERT).
*   Perform hyperparameter tuning for the individual classifiers and the Voting Classifier to potentially further improve performance.
*   Investigate techniques for handling class imbalance if the original six classes were to be included (e.g., oversampling, undersampling, weighted loss functions).
*   Explore deep learning models (e.g., LSTMs, GRUs, Transformers) for sequence-based text classification.
*   Implement a more robust deployment method using a web framework like Flask or FastAPI.
*   Add visualization of text lengths or other text characteristics during EDA.

---

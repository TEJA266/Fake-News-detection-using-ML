# 📰 Fake and Real News Detection

**Misinformation** is a growing challenge in today’s digital age, and distinguishing **fake news** from **real news** has become essential. This project aims to build a machine learning model that classifies news articles as either **real** or **fake** using natural language processing (NLP) techniques.

The dataset used for this project is sourced from Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). It contains around **45,000 news articles** balanced between real and fake news.

---

## 🔍 Project Overview

This project follows a structured approach to detecting fake news using various NLP and machine learning techniques:

- **Data Import and Exploration**  
  - Loaded two datasets containing real and fake news articles (~22,000 records each).
  
- **Data Preprocessing**  
  - Combined datasets and created target labels (`1` for Real, `0` for Fake).
  - Handled missing values and standardized formatting.
  - Merged `title` and `text` columns to improve context for the model.

- **Text Cleaning & Preprocessing**  
  - Removed special characters, stop words, and applied lowercasing.
  - Applied **Stemming** and **Lemmatization** for text normalization.

- **Visualization**  
  - Created **WordClouds** for visualizing frequent terms in real and fake news.

- **Feature Engineering**  
  - Used **CountVectorizer** and **TF-IDF Vectorizer** for text representation.
  - Applied these techniques to both stemmed and lemmatized texts for comparison.

- **Model Development**  
  - Trained various machine learning models:  
    - Logistic Regression  
    - Support Vector Machine (SVM)  
    - Random Forest  
  - Built an **LSTM (Long Short-Term Memory)** model using One-Hot Encoding for deep learning-based classification.

- **Model Evaluation**  
  - Evaluated models using accuracy, precision, recall, and F1-score.

- **Conclusion**  
  - Identified the best-performing models for fake news detection.
  - Analyzed patterns common in real vs. fake news articles using visualizations.

---

## 💾 Dataset Information

The dataset includes two CSV files:

- `True.csv`: Contains **real news articles**  
- `Fake.csv`: Contains **fake news articles**  

Each file consists of:  
- `title`: Headline of the news article  
- `text`: Content of the article  
- `subject`: News category (e.g., politics, world news)  
- `date`: Publication date  

---

## 🛠️ Technologies & Libraries Used

- **Programming Language:** Python  
- **Libraries:**  
  - Data Handling: `Pandas`, `NumPy`  
  - Visualization: `Matplotlib`, `Seaborn`, `WordCloud`  
  - NLP: `NLTK`, `Scikit-learn`  
  - Deep Learning: `TensorFlow`, `Keras`  

---

## 🚀 How to Use This Project


### 📥 Installation Steps

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/fake-and-real-news-detection.git
   cd fake-and-real-news-detection

## 📊 Results
Machine Learning Models: CountVectorizer and TF-IDF delivered high accuracy for text classification.
LSTM Model: Improved classification performance by learning sequential patterns in text data.
Visualization: WordClouds highlighted frequent terms in real and fake news articles.


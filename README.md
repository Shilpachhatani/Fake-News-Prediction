# 📰 Fake News Prediction

## 📌 Project Overview & Task Objective

This notebook focuses on developing a machine learning model to classify news articles as real or fake based on their textual content, utilizing Natural Language Processing (NLP) techniques. The primary objectives include cleaning and preprocessing text data, converting it into a numerical format using TF-IDF, training classification models (Logistic Regression and Random Forest), and evaluating their performance using metrics such as Accuracy, Precision, Recall, and F1-score.

## 📂 Dataset Information

The project typically utilizes a dataset containing news articles labeled as either 'real' or 'fake'. This dataset would include columns for the text content of the news article and a corresponding label indicating its authenticity. Common datasets for this task include `Fake.csv` and `True.csv` or similar structures.

**Key Aspects:**
- Textual data requiring extensive preprocessing.
- Binary classification problem (real vs. fake news).
- Focus on NLP techniques for feature extraction.

## ✨ Features

- Data loading and initial inspection.
- Text cleaning and preprocessing (e.g., lowercasing, removing special characters, stemming).
- Feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency).
- Training and evaluating various classification models (Logistic Regression, Random Forest Classifier).
- Model evaluation using Accuracy, Confusion Matrix, Classification Report (Precision, Recall, F1-score).

## 🛠️ Installation

To run this notebook locally, you will need Python installed along with the following libraries. You can install them using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

Additionally, you might need to download NLTK data:
```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

## 🚀 Approach

My approach to fake news prediction involved the following steps:

- **Library Import**: Imported essential Python libraries for data manipulation (pandas, numpy), text processing (re, nltk), visualization (matplotlib, seaborn), and machine learning (sklearn).
  
- **Data Loading and Combination**: Loaded the real and fake news datasets and combined them into a single DataFrame for unified processing.

- **Data Cleaning and Preprocessing**:
  - Performed text cleaning operations such as converting text to lowercase, removing non-alphabetic characters, and applying stemming using `PorterStemmer`.
  - Tokenized the text and removed stopwords.
    
- **Feature Extraction**: Converted the cleaned text data into numerical features using `TfidfVectorizer`.
  
- **Model Training and Testing**:
  - Split the dataset into training and testing sets.
  - Trained Logistic Regression and Random Forest Classifier models on the TF-IDF features.

- **Model Evaluation**: Evaluated the trained models using accuracy score, confusion matrices, and classification reports to assess their performance in distinguishing between real and fake news.

## 🧰 Technologies Used
- P Y T H O N
- P A N D A S
- N U M P Y
- R E
- N L T K
- M A T P L O T L I B
- S E A B O R N
- S C I K I T - L E A R N

## 📉 Visualizations
### Logistic Regression - Confusion Matrix
![image](https://github.com/user-attachments/assets/0ffb6ba4-b83e-46c1-97fb-10c010f45965)

**Insights:**
- Accuracy: 98.88%
- Fake News: 4585 correct, 65 missed
- Real News: 4294 correct, 36 wrongly marked as fake
- Total mistakes: Only 101 out of 8980
- Precision: 99% – Predictions are mostly correct
- Recall: 99% – Very few fake/real articles missed
- F1-Score: 99% – Excellent balance and consistency

#### Random Forest - Confusion Matrix
![image](https://github.com/user-attachments/assets/39f36adb-7a6a-4a8f-8057-4ba7ecba8b59)

**Insights:**
- Accuracy: 99.65%
- Fake News: 4635 correct, 15 missed
- Real News: 4314 correct, 16 wrongly marked as fake
- Total mistakes: Only 31 out of 8980
- Precision: 100% – Predictions are almost always correct
- Recall: 100% – Very few fake/real articles missed
- F1-Score: 100% – Perfect balance and consistency

## 📊 Results and Insights

### Key Insights:
  - Text preprocessing and TF-IDF vectorization are crucial steps for preparing textual data for machine learning models.
  - Both Logistic Regression and Random Forest models can achieve high accuracy in classifying fake news, demonstrating the effectiveness of NLP in this domain.
  - Evaluation metrics like precision, recall, and F1-score provide a comprehensive understanding of model performance, especially in identifying false positives and false negatives.
    
### Final Outcome:
  - This project successfully demonstrates a complete pipeline for fake news detection using machine learning and NLP.
  - The models developed can serve as a baseline for more advanced fake news detection systems.

## 🧪 Usage

```bash
# 1. Clone the repository 
git clone https://github.com/AhsanNFt/Fake_News_Prediction.git

# 2. Navigate to the project directory
cd Fake_News_Prediction

# 3. Open the notebook
jupyter notebook Fake_News_Prediction.ipynb

```

## 🤝 Contributing

Contributions are welcome! If you have any suggestions or improvements, please open an issue or submit a pull request.

## 📬 Contact

For questions or collaboration:
- GitHub: `AhsanNFt`
- Email: syedahsan0991@gmail.com.



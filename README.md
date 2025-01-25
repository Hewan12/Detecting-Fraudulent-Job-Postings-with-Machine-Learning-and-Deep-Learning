# Detecting Fraudulent Job Postings with Machine Learning and Deep Learning

## 📜 Project Overview

This project focuses on detecting fraudulent job postings using various machine learning (ML) and deep learning (DL) techniques. Scammers often post enticing job descriptions to steal sensitive personal information from applicants. By analyzing a Kaggle dataset containing real and fake job postings, this project builds robust classification models to distinguish legitimate job posts from fake ones.

## 🔍 Dataset Overview

- **Source**: [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size**: 18,000 job descriptions

### Features:
- Textual information (e.g., job description)
- Meta-information (e.g., location, employment type, required experience)

### Target Variable:
- **fraudulent** (1 = Fake job post, 0 = Real job post)

### Challenges:
- **Imbalanced dataset**: ~800 fake job posts vs. ~17,200 real ones
- **Text data variability**: Job descriptions vary in length and quality

## 🚀 Project Pipeline

### Data Understanding:
- Loaded the dataset and inspected its structure, data types, and missing values.
- Analyzed class distribution to address imbalance.

### Data Cleaning and Preprocessing:
- Cleaned text data by:
  - Removing HTML tags, special characters, and stop words.
  - Converting text to lowercase.
- Encoded categorical features (e.g., location, employment type) using `LabelEncoder`.
- Scaled numerical features using `StandardScaler` for improved model performance.
- Transformed job descriptions into numerical features using `TF-IDF` with a limit of 5,000 features.

### Feature Engineering:
- Combined TF-IDF features with scaled meta-information to create the final feature set.

### Model Building:
- Split data into training (80%) and testing (20%) sets using `train_test_split` with stratification.
- Implemented the following models:
  - **Support Vector Machine (SVM)**: Achieved strong precision and recall for smaller feature sets.
  - **Random Forest**: Baseline model, robust but prone to overfitting on imbalanced data.
  - **Deep Neural Network (DNN)**: Multilayer perceptron with ReLU activation and dropout for regularization.
  - **LSTM**: Captured sequential patterns in job descriptions using word embeddings and a bidirectional architecture.

### Evaluation:
#### Metrics used:
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- ROC-AUC Score

#### Visualizations:
- Heatmaps for confusion matrices
- Word clouds for fraudulent vs. real job descriptions
- Performance comparison charts for all models

## 📈 Results

| Model           | Precision | Recall | F1-score | ROC-AUC |
|-----------------|-----------|--------|----------|---------|
| **SVM**         | 0.89      | 0.81   | 0.85     | 0.87    |
| **Random Forest** | 0.92      | 0.79   | 0.85     | 0.90    |
| **DNN**         | 0.91      | 0.85   | 0.88     | 0.92    |
| **LSTM**        | 0.94      | 0.87   | 0.90     | 0.94    |

### Insights from Word Clouds:
- Fraudulent job descriptions often use vague and overly enticing terms (e.g., "easy money," "work from home").
- Legitimate posts include industry-specific jargon and professional terminology.

## 🛠️ Tools and Technologies

- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: pandas, numpy
  - Visualization: matplotlib, seaborn, wordcloud
  - Machine Learning: sklearn
  - Deep Learning: tensorflow, keras
- **IDE**: Jupyter Notebook

## 📌 Future Improvements

- Explore transformer-based models like **BERT** for improved text classification.
- Implement advanced techniques to handle class imbalance, such as:
  - **Oversampling** (e.g., SMOTE)
  - **Cost-sensitive learning**
- Perform hyperparameter tuning for **SVM**, **Random Forest**, and **neural networks**.
- Incorporate external datasets to enhance model generalization.

🤝 Contributing

Contributions are welcome! Feel free to fork this repository, open issues, or submit pull requests to enhance the project.

📧 Contact

For any questions or suggestions, please reach out:

    Name: Hewan Leul
    LinkedIn: www.linkedin.com/in/hewanleul12

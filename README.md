# 🎬 Sentiment Analysis on IMDB Movie Reviews

A machine learning project that analyzes movie reviews and predicts whether they are **Positive** or **Negative** using Natural Language Processing (NLP).

---

## 📌 Project Overview

This project uses the **IMDB dataset** (50,000 real movie reviews) to train a sentiment analysis model. It converts text reviews into numbers using **TF-IDF Vectorization** and classifies them using **Logistic Regression**.

---

## 🛠️ Technologies Used

- Python 3.x
- Scikit-learn (Machine Learning)
- HuggingFace Datasets (IMDB data)
- Seaborn & Matplotlib (Visualization)
- TF-IDF Vectorizer (Text Processing)

---

## 📂 Project Structure

```
sentiment_project/
│
├── train.py              # Main Python script
├── requirements.txt      # Required libraries
└── README.md             # Project documentation
```

---

## ⚙️ How to Run This Project

### Step 1: Clone the repository
```bash
git clone https://github.com/sarthak25bai10247-eng/AI-ML-PROJECT.git
cd sentiment_project
```

### Step 2: Install required libraries
```bash
pip install -r requirements.txt
```

### Step 3: Run the project
```bash
python train.py
```

---

## 📊 Expected Output

```
Step 1: Loading dataset...
Step 2: Preparing data...
Step 3: Converting text to numbers...
Step 4: Training the model...
Step 5: Testing the model...

--- RESULTS ---
              precision    recall  f1-score
    Negative       0.89      0.88      0.88
    Positive       0.88      0.89      0.89
```

A **Confusion Matrix chart** will also pop up showing model accuracy visually.

---

## 🧠 How It Works

1. **Downloads** 50,000 IMDB movie reviews (positive & negative)
2. **Converts** text into numbers using TF-IDF Vectorization
3. **Trains** a Logistic Regression model on 25,000 reviews
4. **Tests** the model on 25,000 unseen reviews
5. **Displays** accuracy report and confusion matrix

---

## 👨‍🎓 University Project

- **Student:** Sarthak Jena
- **Registration Number:** 25BAI10247
- **Project:** Sentiment Analysis using Machine Learning
- **Language:** Python
- **GitHub:** https://github.com/sarthak25bai10247-eng/

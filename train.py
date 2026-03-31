# Step 1: Import the tools we installed
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("Step 1: Loading dataset...")
# Downloads 50,000 real movie reviews from the internet (labelled positive/negative)
dataset = load_dataset("imdb")

print("Step 2: Preparing data...")
# Separate the text (reviews) from the labels (0=negative, 1=positive)
train_texts  = dataset["train"]["text"]
train_labels = dataset["train"]["label"]
test_texts   = dataset["test"]["text"]
test_labels  = dataset["test"]["label"]

print("Step 3: Converting text to numbers...")
# Computers can't read words — TF-IDF converts each review into
# a list of numbers representing how important each word is
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
X_train = vectorizer.fit_transform(train_texts)
X_test  = vectorizer.transform(test_texts)

print("Step 4: Training the model...")
# Logistic Regression learns patterns: which words = positive? which = negative?
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train, train_labels)

print("Step 5: Testing the model...")
# Run the model on reviews it has NEVER seen before
preds = model.predict(X_test)

print("\n--- RESULTS ---")
print(classification_report(test_labels, preds, target_names=["Negative", "Positive"]))

print("Step 6: Showing confusion matrix chart...")
# Visual chart: how many did it get right vs wrong?
cm = confusion_matrix(test_labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
plt.title("Confusion Matrix — How accurate is our model?")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\nDone! Your first AI sentiment model is working.")
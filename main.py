import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from lime.lime_text import LimeTextExplainer
from wordcloud import WordCloud
from sklearn.model_selection import StratifiedKFold



# Load the dataset with a specified encoding
df = pd.read_csv('train.csv', encoding='latin1')  # You can try 'ISO-8859-1' or 'cp1252' as well


# 1. Data Exploration
print(df.info())
print(df.describe())
print(df.head())
print(df['sentiment'].value_counts())

# 2. Data Preprocessing
# Placeholder for text preprocessing
def preprocess_text(text):
    # Check if the value is not NaN before applying lower()
    if pd.notna(text):
        processed_text = text.lower()
        # Add your additional preprocessing steps here
    else:
        processed_text = ""  # or any other suitable handling for NaN values
    return processed_text

# 2. Data Preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)


# 3. Exploratory Data Analysis (EDA)
plt.figure(figsize=(8, 6))
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'yellow', 'red'])
plt.title('Distribution of Sentiment Labels')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='sentiment', hue='some_additional_feature', data=df)
plt.title('Distribution of Sentiment Labels by Additional Feature')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# 4. Text Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X = vectorizer.fit_transform(df['processed_text'])
le = LabelEncoder()
y = le.fit_transform(df['sentiment'])

# Try Word2Vec for text vectorization
w2v_model = Word2Vec(sentences=df['processed_text'].apply(lambda x: x.split()), vector_size=100, window=5, min_count=1, workers=4)
X_w2v = np.array([np.mean([w2v_model.wv[word] for word in words.split() if word in w2v_model.wv] or [np.zeros(100)], axis=0) for words in df['processed_text']])

# 5. Model Selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)

# Random Forest
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Evaluate the performance
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Naive Bayes Precision:", precision_score(y_test, y_pred_nb, average='weighted'))
print("Naive Bayes Recall:", recall_score(y_test, y_pred_nb, average='weighted'))
print("Naive Bayes F1 Score:", f1_score(y_test, y_pred_nb, average='weighted'))

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("Random Forest Recall:", recall_score(y_test, y_pred_rf, average='weighted'))
print("Random Forest F1 Score:", f1_score(y_test, y_pred_rf, average='weighted'))

# 6. Hyperparameter Tuning (Random Forest)
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search_rf = RandomizedSearchCV(model_rf, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, scoring='accuracy')
random_search_rf.fit(X_train, y_train)
best_model_rf = random_search_rf.best_estimator_

# 7. Cross-Validation
cv_scores_nb = cross_val_score(model_nb, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
cv_scores_rf = cross_val_score(best_model_rf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print("Cross-validation scores (Naive Bayes):", cv_scores_nb)
print("Mean CV accuracy (Naive Bayes):", np.mean(cv_scores_nb))
print("Cross-validation scores (Random Forest):", cv_scores_rf)
print("Mean CV accuracy (Random Forest):", np.mean(cv_scores_rf))

# 8. Model Interpretability (using LIME)
explainer = LimeTextExplainer()
idx = 0  # Index of the instance to explain
exp = explainer.explain_instance(df['processed_text'].iloc[idx], model_nb.predict_proba, num_features=10)
print("LIME explanation (Naive Bayes):", exp.as_list())

# 9. Evaluation Metrics
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix (Naive Bayes):\n", conf_matrix_nb)
print("Confusion Matrix (Random Forest):\n", conf_matrix_rf)

# ROC-AUC for Random Forest
y_prob_rf = best_model_rf.predict_proba(X_test)
macro_roc_auc_rf = roc_auc_score(y_test, y_prob_rf, multi_class="ovr", average="macro")
print("ROC-AUC (Random Forest):", macro_roc_auc_rf)

# ROC curves for each class
fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()
for i in range(len(le.classes_)):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test, y_prob_rf[:, i], pos_label=i)
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

plt.figure(figsize=(10, 6))
for i in range(len(le.classes_)):
    plt.plot(fpr_rf[i], tpr_rf[i], label=f'Class {i} (AUC = {roc_auc_rf[i]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for each Class (Random Forest)')
plt.legend()
plt.show()

# 10. Additional Features
# Word Cloud for Positive and Negative Sentiments
positive_words = ' '.join(df[df['sentiment'] == 'positive']['processed_text'])
negative_words = ' '.join(df[df['sentiment'] == 'negative']['processed_text'])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_words)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Word Cloud for Positive Sentiment')

plt.subplot(1, 2, 2)
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_words)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Word Cloud for Negative Sentiment')

plt.show()

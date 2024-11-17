import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Bước 1: Đọc tệp dữ liệu
data = pd.read_csv('/home/nquang/BTL-AI/Spam-Email-Detection-using-MultinomialNB/spam_ham_dataset.csv')

# Bước 2: Chuẩn bị dữ liệu và nhãn
X = data['text']  # Email text
y = data['label'].apply(lambda x: 1 if x == 'spam' else 0)  # Đổi nhãn thành 1 (spam) và 0 (ham)

# Bước 3: Chia tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 4: Biến đổi văn bản thành vector đặc trưng
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Bước 5: Huấn luyện mô hình Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Bước 6: Lưu mô hình và vectorizer
joblib.dump(model, '/home/nquang/BTL-AI/Spam-Email-Detection-using-MultinomialNB/checkpoints/spam_detection_model.pkl')
joblib.dump(vectorizer, '/home/nquang/BTL-AI/Spam-Email-Detection-using-MultinomialNB/checkpoints/count_vectorizer.pkl')
print("Model and vectorizer saved successfully.")

import streamlit as st
import joblib
import re
import time
import pandas as pd
import html

st.set_page_config(page_title="HỆ THỐNG KIỂM TRA EMAIL SPAM", page_icon="✉️")

@st.cache_resource(ttl=3600)
def load_model():
    learn_inf = joblib.load('/home/nquang/BTL-AI/Spam-Email-Detection-using-MultinomialNB/checkpoints/spam_detection_model.pkl')
    vectorizer = joblib.load('/home/nquang/BTL-AI/Spam-Email-Detection-using-MultinomialNB/checkpoints/count_vectorizer.pkl')
    return learn_inf, vectorizer

@st.cache_resource
def load_spam_keywords():
    data = pd.read_csv('/home/nquang/BTL-AI/Spam-Email-Detection-using-MultinomialNB/spam_ham_dataset.csv')
    spam_emails = data[data['label'] == 'spam']['text']
    spam_keywords = set()
    for email in spam_emails:
        spam_keywords.update(email.lower().split())
    spam_keywords = [word for word in spam_keywords if len(word) > 3]
    return spam_keywords

def classify_email(model, vectorizer, email):
    prediction = model.predict(vectorizer.transform([email]))[0]
    return prediction

def highlight_keywords(text, keywords):
    text = html.escape(text)
    text = text.replace('\n', '<br>')
    words = text.split(' ')
    
    highlighted_words = []
    for word in words:
        word_lower = word.lower()
        word_clean = re.sub(r'[^\w\s]', '', word_lower)
        
        if word_clean in [k.lower() for k in keywords]:
            highlighted_words.append(f'<mark style="background-color: #FF9999;">{word}</mark>')
        else:
            highlighted_words.append(word)
    
    return ' '.join(highlighted_words)

def get_email_statistics(email_text, spam_keywords):
    word_count = len(email_text.split())
    keyword_occurrences = sum(len(re.findall(rf'\b{re.escape(keyword)}\b', email_text, flags=re.IGNORECASE)) for keyword in spam_keywords)
    return word_count, keyword_occurrences

def main():
    st.title("HỆ THỐNG KIỂM TRA EMAIL SPAM")
    
    # Cập nhật CSS với theme tối
    st.markdown("""
        <style>
        .email-container {
            background-color: #0e1117;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #464c55;
            margin: 10px 0;
            color: #fafafa;
        }
        mark {
            padding: 2px 4px;
            border-radius: 3px;
            background-color: #FF9999;
            color: #0e1117;
        }
        </style>
    """, unsafe_allow_html=True)
    
    output = st.empty()
    status_bar = st.empty()

    with st.spinner('Đang tải trang web...'):
        user_input = st.text_area('Nhập nội dung email:', '', placeholder='Chúc mừng!! Bạn đã thắng Rs. 100000. \nNhấn vào đây để nhận!!')
    
    spam_keywords = load_spam_keywords()
    
    if st.button("Kiểm tra Spam"):
        output.empty()
        status_bar.empty()
        if user_input:
            with status_bar.status("Đang tải mô hình...", expanded=True) as status:
                model, vectorizer = load_model()
                time.sleep(2)

                status.update(label="Đang phân tích email...", state="running", expanded=True)
                time.sleep(2)

                status.update(label="Đang kiểm tra Spam...", state="running", expanded=True)
                prediction = classify_email(model, vectorizer, user_input)
                time.sleep(2)

                status.update(label="Kiểm tra hoàn tất!", state="complete", expanded=False)

            status_bar.empty()
            if prediction == 1:
                output.error('Phát hiện Spam!')

                st.subheader("Từ khóa nổi bật trong email")
                highlighted_text = highlight_keywords(user_input, spam_keywords)
                st.markdown(f'<div class="email-container">{highlighted_text}</div>', unsafe_allow_html=True)

                word_count, keyword_occurrences = get_email_statistics(user_input, spam_keywords)
                st.subheader("Thống kê nội dung email")
                st.write(f"- Tổng số từ: {word_count}")
                st.write(f"- Số từ khóa spam: {keyword_occurrences}")

                st.subheader("Gợi ý hành động")
                st.warning("Cân nhắc đánh dấu email này là spam hoặc xóa đi.")
                
            else:
                output.success('Không phải Spam')
                st.subheader("Gợi ý hành động")
                st.info("Email này có vẻ an toàn, nhưng hãy cẩn thận với các liên kết không rõ nguồn gốc.")

        else:
            output.warning("Vui lòng nhập nội dung để kiểm tra !!")

if __name__ == "__main__":
    main()

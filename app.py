import streamlit as st
import joblib
import re
import time
import pandas as pd
import html
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="HỆ THỐNG KIỂM TRA EMAIL SPAM", page_icon="✉️", layout="wide")

@st.cache_resource(ttl=3600)
def load_model():
    learn_inf = joblib.load('/home/nquang/1_BTL-AI/spam-email-ai-ver2/checkpoints/spam_detection_model.pkl')
    vectorizer = joblib.load('/home/nquang/1_BTL-AI/spam-email-ai-ver2/checkpoints/count_vectorizer.pkl')
    return learn_inf, vectorizer

@st.cache_resource
def load_spam_keywords():
    data = pd.read_csv('/home/nquang/1_BTL-AI/spam-email-ai-ver2/spam_ham_dataset.csv')
    spam_emails = data[data['label'] == 'spam']['text']
    spam_keywords = set()
    for email in spam_emails:
        spam_keywords.update(email.lower().split())
    spam_keywords = [word for word in spam_keywords if len(word) > 3]
    return spam_keywords

def save_to_history(email_text, is_spam, spam_probability, word_count, spam_count):
    # Initialize history in session state if it doesn't exist
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Add new entry to history
    st.session_state.history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'email_text': email_text,
        'is_spam': is_spam,
        'spam_probability': spam_probability,
        'word_count': word_count,
        'spam_count': spam_count
    })

def classify_email(model, vectorizer, email):
    prediction = model.predict(vectorizer.transform([email]))[0]
    proba = model.predict_proba(vectorizer.transform([email]))[0]
    return prediction, proba[1]

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
    words = email_text.split()
    word_count = len(words)
    
    spam_word_counts = {}
    total_spam_words = 0
    
    for word in words:
        word_clean = re.sub(r'[^\w\s]', '', word.lower())
        if word_clean in [k.lower() for k in spam_keywords]:
            spam_word_counts[word_clean] = spam_word_counts.get(word_clean, 0) + 1
            total_spam_words += 1
    
    return word_count, total_spam_words, spam_word_counts

def create_distribution_chart(word_count, spam_count):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Từ thường', 'Từ khóa spam'],
        y=[word_count - spam_count, spam_count],
        marker_color=['#00CC96', '#EF553B']
    ))
    
    fig.update_layout(
        title='Phân bố từ khóa spam trong email',
        yaxis_title='Số lượng từ',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    return fig

def create_trend_chart(history):
    if not history:
        return None
        
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['spam_probability'],
        mode='lines+markers',
        name='Xác suất spam',
        line=dict(color='#EF553B')
    ))
    
    fig.update_layout(
        title='Xu hướng phát hiện spam theo thời gian',
        xaxis_title='Thời gian',
        yaxis_title='Xác suất spam',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def main():
    st.title("HỆ THỐNG KIỂM TRA EMAIL SPAM")
    
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

    tabs = st.tabs(["Kiểm tra Email", "Phân tích chi tiết", "Lịch sử"])
    
    # Khởi tạo các biến để sử dụng trong nhiều tabs
    spam_keywords = load_spam_keywords()
    word_count = 0
    spam_count = 0
    spam_words = {}
    prediction = None
    spam_probability = 0
    
    with tabs[0]:
        output = st.empty()
        status_bar = st.empty()

        user_input = st.text_area('Nhập nội dung email:', '', 
                                 placeholder='Chúc mừng!! Bạn đã thắng Rs. 100000. \nNhấn vào đây để nhận!!')
        
        if st.button("Kiểm tra Spam"):
            if user_input:
                with status_bar.status("Đang xử lý...", expanded=True) as status:
                    model, vectorizer = load_model()
                    prediction, spam_probability = classify_email(model, vectorizer, user_input)
                    word_count, spam_count, spam_words = get_email_statistics(user_input, spam_keywords)
                    
                    if prediction == 1:
                        output.error('⚠️ Phát hiện Spam!')
                        st.progress(spam_probability, text=f"Xác suất spam: {spam_probability:.1%}")
                        
                        st.subheader("Từ khóa đáng ngờ")
                        highlighted_text = highlight_keywords(user_input, spam_keywords)
                        st.markdown(f'<div class="email-container">{highlighted_text}</div>', 
                                  unsafe_allow_html=True)
                        
                        st.subheader("Gợi ý hành động")
                        if spam_probability > 0.8:
                            st.error("Email có độ nguy hiểm cao! Nên xóa ngay lập tức và chặn người gửi.")
                        else:
                            st.warning("Cân nhắc đánh dấu email này là spam hoặc xóa đi.")
                    else:
                        output.success('✅ Không phải Spam')
                        st.progress(1 - spam_probability, text=f"Độ tin cậy: {(1-spam_probability):.1%}")
                        st.subheader("Gợi ý hành động")
                        st.info("Email này có vẻ an toàn, nhưng hãy cẩn thận với các liên kết không rõ nguồn gốc.")
                
                # Lưu kết quả vào lịch sử session
                if st.session_state.get('last_checked') != user_input:
                    save_to_history(user_input, prediction == 1, spam_probability, word_count, spam_count)
                    st.session_state['last_checked'] = user_input
            else:
                output.warning("Vui lòng nhập nội dung để kiểm tra !!")

    with tabs[1]:
        if user_input and prediction is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Thống kê nội dung")
                st.metric("Tổng số từ", word_count)
                st.metric("Số từ khóa spam", spam_count)
                st.metric("Tỷ lệ từ khóa spam", f"{(spam_count/word_count*100):.1f}%" if word_count > 0 else "0%")
            
            with col2:
                st.subheader("Phân bố từ trong email")
                fig = create_distribution_chart(word_count, spam_count)
                st.plotly_chart(fig, use_container_width=True)
            
            if spam_count > 0:
                st.subheader("Chi tiết từ khóa spam")
                spam_df = pd.DataFrame(list(spam_words.items()), 
                                     columns=['Từ khóa', 'Số lần xuất hiện'])
                st.dataframe(spam_df.sort_values('Số lần xuất hiện', ascending=False))
        else:
            st.info("Hãy nhập email ở tab 'Kiểm tra Email' để xem phân tích chi tiết.")

    with tabs[2]:
        st.subheader("Lịch sử kiểm tra email")
        
        # Lấy lịch sử từ session state
        history = st.session_state.get('history', [])
        if not history:
            st.info("Chưa có email nào được kiểm tra.")
        else:
            # Biểu đồ xu hướng
            st.subheader("Xu hướng phát hiện spam")
            trend_chart = create_trend_chart(history)
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
            
            # Thống kê
            total_emails = len(history)
            spam_emails = sum(1 for entry in history if entry['is_spam'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng số email đã kiểm tra", total_emails)
            with col2:
                st.metric("Số email spam", spam_emails)
            with col3:
                st.metric("Tỷ lệ spam", f"{(spam_emails/total_emails*100):.1f}%" if total_emails > 0 else "0%")
            
            # Bảng lịch sử
            st.subheader("Chi tiết lịch sử kiểm tra")
            history_df = pd.DataFrame(history)
            history_df['Thời gian'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            history_df['Kết quả'] = history_df['is_spam'].map({True: '❌ Spam', False: '✅ An toàn'})
            history_df['Xác suất spam'] = history_df['spam_probability'].map('{:.1%}'.format)
            
            display_df = history_df[['Thời gian', 'email_text', 'Kết quả', 'Xác suất spam', 'word_count', 'spam_count']]
            display_df.columns = ['Thời gian', 'Nội dung email', 'Kết quả', 'Xác suất spam', 'Số từ', 'Số từ spam']
            st.dataframe(display_df.sort_values('Thời gian', ascending=False), use_container_width=True)

            # Nút xóa lịch sử
            if st.button("Xóa lịch sử"):
                st.session_state.history = []
                st.success("Đã xóa toàn bộ lịch sử!")
                st.rerun()

if __name__ == "__main__":
    main()
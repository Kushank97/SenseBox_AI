import streamlit as st
import joblib
import pandas as pd
import os
from streamlit_lottie import st_lottie
import json
from typing import Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_DIR = os.path.join(BASE_DIR, "models")


spam_model = joblib.load(os.path.join(MODEL_DIR, "spam_classifier.pkl"))
language_model = joblib.load(os.path.join(MODEL_DIR, "lang_det.pkl"))
news_model = joblib.load(os.path.join(MODEL_DIR, "news_cat.pkl"))
review_model = joblib.load(os.path.join(MODEL_DIR, "review.pkl"))

def load_lottiefile(filepath: str) -> Optional[dict]:
    """Load Lottie animation file"""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None


animation_spam = load_lottiefile("spam_animation.json") if os.path.exists("spam_animation.json") else None
animation_language = load_lottiefile("language_animation.json") if os.path.exists("language_animation.json") else None
animation_sentiment = load_lottiefile("sentiment_animation.json") if os.path.exists("sentiment_animation.json") else None
animation_news = load_lottiefile("news_animation.json") if os.path.exists("news_animation.json") else None


def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        [data-testid="stSidebar"] {
            background-color: #000000 !important;
            border-right: 1px solid #333333;
        }
        .sidebar .sidebar-content {
            background-color: #000000;
        }
        .main {
            background-color: rgba(25, 25, 50, 0.9) !important;
            padding: 2.5rem;
            border-radius: 15px;
            margin: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }
        p, div {
            color: #e6e6e6 !important;
        }
        .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj {
            background-color: transparent !important;
        }
        .stTextInput input, .stTextArea textarea {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
        }
        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
        }
        .stAlert .st-b7 {
            background-color: rgba(46, 125, 50, 0.9) !important;
        }
        .dataframe {
            background-color: rgba(30, 30, 60, 0.9) !important;
            color: white !important;
        }
        @media (max-width: 768px) {
            .main {
                padding: 1.5rem;
                margin: 0.5rem;
            }
        }
        .about-me {
            color: #ffffff !important;
            line-height: 1.6;
        }
        .highlight {
            color: #4CAF50;
            font-weight: bold;
        }
        .title-text {
            color: #4CAF50 !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
            animation: fadeIn 1.5s ease-in-out;
        }
        .model-info {
            background-color: rgba(40, 40, 80, 0.7) !important;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        .model-info:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        }
        .portfolio-tab {
            background: linear-gradient(to bottom, #1a1a2e, #16213e);
            padding: 2rem;
            border-radius: 15px;
            margin: -1rem;
            animation: slideIn 1s ease-out;
        }
        .portfolio-project {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            border-left: 4px solid #4CAF50;
            transition: all 0.3s ease;
        }
        .portfolio-project:hover {
            transform: scale(1.01);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.2);
        }
        .project-title {
            color: #4CAF50 !important;
            margin-bottom: 1rem !important;
        }
        .project-image {
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        .project-image:hover {
            transform: scale(1.03);
        }
        .portfolio-tab p, .portfolio-tab li {
            color: #ffffff !important;
        }
        .portfolio-tab h1, .portfolio-tab h2, .portfolio-tab h3 {
            color: #4CAF50 !important;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateY(50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .tab-content {
            animation: fadeIn 0.8s ease-out;
        }
        .success-message {
            animation: bounce 0.5s ease;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    


st.set_page_config(
    page_title="Sense Box AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


set_background()

with st.sidebar:
    st.markdown("""
    <style>
    /* Add this new CSS for file uploader */
    .stFileUploader > div > div {
        background-color: #000000 !important;
        border-radius: 8px;
        padding: 1rem;
    }
    .stFileUploader > div > div:hover {
        border-color: #4CAF50 !important;
    }
    .stFileUploader > label > div > p {
        color: white !important;
    }
    
    /* Animation for sidebar image */
    .sidebar-image {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    
    /* Animation for project cards */
    .project-card-animation {
        transition: all 0.5s ease;
    }
    .project-card-animation:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 25px rgba(76, 175, 80, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if os.path.exists("kushank.png"):
        st.image("kushank.png", caption="Sense Box AI", use_container_width=True)
    else:
        st.warning("⚠️ kushank.png not found in project folder.")

    with st.expander("🧑‍💻 About Me"):
        st.markdown("""
        <div class="about-me">
        <span class="highlight">Kushank Sharma</span> — Data enthusiast skilled in Python, SQL, and Machine Learning. 
        Experienced in Data Analysis, NLP, sentiment analysis, and classification. Passionate about turning raw data into insights 
        and building real-world solutions.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("ℹ️ About Sense Box AI"):
        st.markdown("""
        <div class="about-me">
        Sense Box AI is a machine learning web app that uses NLP techniques to classify and analyze text. 
        Built with NLTK, scikit-learn, and Streamlit, it features four predictive models for tasks like 
        spam detection, sentiment analysis, and language detection.
        
        The app uses:
        - TF-IDF vectorization
        - Algorithms like Logistic Regression and Multinomial Naive Bayes
        - Tools like Pipeline, Joblib, and imblearn 
        
        for efficient training and deployment.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📞 Contact Us"):
        st.markdown("### Get in Touch")
        st.link_button("📧 Email", "mailto:kushanksharma97@gmail.com")
        st.link_button("💻 GitHub", "https://github.com/Kushank97")
        st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/kushank-sharma-72bb86296")


st.markdown('<h1 class="title-text">🎯 Sense Box AI: Market Sentiment Engine</h1>', unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Spam Classifier",
    "🗣️ Language Detection", 
    "🍽️ Food Review Sentiment",
    "📰 News Classification",
    "📊 Kushank Data Analyst Portfolio"
])

def read_uploaded_file(uploaded_file):
    """Helper function to read both CSV and TXT files"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:  # TXT file
            content = uploaded_file.read().decode('utf-8')
            return pd.DataFrame({'text': content.split('\n')})
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None


with tab1:
    st.header("📩 Spam Classifier")
    
    if animation_spam:
        st_lottie(animation_spam, height=200, key="spam_animation")
    
    with st.container():
        st.markdown("""
        <div class="model-info">
        <h4>Model Description</h4>
        <p>This model classifies text messages as either spam or not spam using machine learning. 
        Trained on labeled data, it uses TF-IDF vectorization and algorithms like Logistic Regression 
        to detect promotional or harmful content, helping filter unwanted messages effectively.</p>
        </div>
        """, unsafe_allow_html=True)

    msg = st.text_input("Enter a message to classify")
    if st.button("Detect Spam"):
        pred = spam_model.predict([msg])
        if pred[0] == 0:
            st.success("❌ Spam Detected!", icon="⚠️")
            st.image("spams.webp", width=300)
        else:
            st.success("✅ Not Spam!", icon="👍")
            st.image("tick.jpg", width=300)

    uploaded_file = st.file_uploader("Upload a file (CSV or TXT)", type=["csv", "txt"])
    if uploaded_file:
        df = read_uploaded_file(uploaded_file)
        if df is not None:
            if len(df.columns) > 1:
                st.warning("Using first column as messages")
            df_spam = df[[df.columns[0]]].copy()
            df_spam.columns = ["Msg"]
            df_spam.index = range(1, df_spam.shape[0] + 1)
            df_spam["Prediction"] = spam_model.predict(df_spam["Msg"])
            df_spam["Prediction"] = df_spam["Prediction"].map({0: 'Spam', 1: 'Not Spam'})
            st.dataframe(df_spam)


with tab2:
    st.header("🌐 Language Detection")
    
    if animation_language:
        st_lottie(animation_language, height=200, key="language_animation")
    
    with st.container():
        st.markdown("""
        <div class="model-info">
        <h4>Model Description</h4>
        <p>This model identifies the language of any given text input. It analyzes character 
        and word patterns to distinguish between multiple languages. Useful for multilingual 
        applications, the model combines text preprocessing and classification techniques to 
        deliver fast, reliable results.</p>
        </div>
        """, unsafe_allow_html=True)

    text = st.text_area("Enter text to detect language")
    if st.button("Detect Language"):
        pred = language_model.predict([text])
        st.success(f"🈯 Detected Language: **{pred[0]}**", icon="🌍")

    uploaded_file = st.file_uploader("Upload a file (CSV or TXT)", type=["csv", "txt"], key="lang")
    if uploaded_file:
        df = read_uploaded_file(uploaded_file)
        if df is not None:
            if len(df.columns) > 1 and 'Text' in df.columns:
                df_lang = df[['Text']].copy()
            elif len(df.columns) > 1:
                st.warning("Using first column as text")
                df_lang = df[[df.columns[0]]].copy()
                df_lang.columns = ['Text']
            else:
                df_lang = df.copy()
                df_lang.columns = ['Text']
            
            df_lang["Language"] = language_model.predict(df_lang["Text"])
            st.dataframe(df_lang)


with tab3:
    st.header("🍽️ Food Review Sentiment")
    
    if animation_sentiment:
        st_lottie(animation_sentiment, height=200, key="sentiment_animation")
    
    with st.container():
        st.markdown("""
        <div class="model-info">
        <h4>Model Description</h4>
        <p>The sentiment analysis model determines whether a given text expresses positive or negative 
        emotions. It applies NLP techniques to understand tone and opinion, commonly used in reviews, 
        feedback, or social media analysis. The model uses TF-IDF and supervised algorithms for 
        accurate predictions.</p>
        </div>
        """, unsafe_allow_html=True)

    review = st.text_area("Enter a food review")
    if st.button("Analyze Sentiment"):
        pred = review_model.predict([review])
        if pred[0] == 0:
            st.success("👎 Negative Feedback", icon="😞")
        else:
            st.success("👍 Positive Feedback", icon="😊")

    uploaded_file = st.file_uploader("Upload a file (CSV or TXT)", type=["csv", "txt"], key="review")
    if uploaded_file:
        df = read_uploaded_file(uploaded_file)
        if df is not None:
            if len(df.columns) > 1 and 'Review' in df.columns:
                df_rev = df[['Review']].copy()
            elif len(df.columns) > 1:
                st.warning("Using first column as reviews")
                df_rev = df[[df.columns[0]]].copy()
                df_rev.columns = ['Review']
            else:
                df_rev = df.copy()
                df_rev.columns = ['Review']
            
            df_rev["Sentiment"] = review_model.predict(df_rev["Review"])
            df_rev["Sentiment"] = df_rev["Sentiment"].map({0: 'Negative Feedback', 1: 'Positive Feedback'})
            st.dataframe(df_rev)


with tab4:
    st.header("📰 News Classification")
    
    if animation_news:
        st_lottie(animation_news, height=200, key="news_animation")
    
    with st.container():
        st.markdown("""
        <div class="model-info">
        <h4>Model Description</h4>
        <p>This model is currently under development to deliver more accurate and reliable category 
        predictions for news articles. I am improving its training data and algorithms to ensure 
        better performance in classifying news into topics like sports, politics, and technology. 
        Stay tuned for updates!</p>
        </div>
        """, unsafe_allow_html=True)

    
        if os.path.exists("under_construction.png"):
            st.image("under_construction.png", width=300)
        else:
            st.warning("⚠️ under_construction.png not found in project folder.")


with tab5:
    
    st.markdown(
        """
        <style>
        .portfolio-tab {
            background: linear-gradient(to bottom, #1a1a2e, #16213e);
            padding: 2rem;
            border-radius: 15px;
            margin: -1rem;
        }
        .portfolio-project {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            border-left: 4px solid #4CAF50;
        }
        .project-title {
            color: #4CAF50 !important;
            margin-bottom: 1rem !important;
        }
        .project-image {
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .portfolio-tab p, .portfolio-tab li {
            color: #ffffff !important;
        }
        .portfolio-tab h1, .portfolio-tab h2, .portfolio-tab h3 {
            color: #4CAF50 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    with st.container():
        st.markdown('<div class="portfolio-tab">', unsafe_allow_html=True)
        
        st.markdown('<h1 style="color:#4CAF50; text-align:center;">📊 Data Analyst Portfolio</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # ===== Project 1 =====
        with st.container():
            st.markdown('<div class="portfolio-project">', unsafe_allow_html=True)
            
            st.markdown('<h3 class="project-title">1. Netflix Content Analysis</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if os.path.exists("netflix.png"):
                    st.image("netflix.png", width=300, use_container_width=True, clamp=True, caption="Netflix Analysis Dashboard")
                else:
                    st.warning("Image not found: netflix.png")
            
            with col2:
                st.markdown("""
                **Project Overview**:  
                Comprehensive analysis of Netflix's content catalog revealing:
                - Content distribution (Movies vs TV Shows)
                - Release trends and seasonality patterns
                - Geographic production hubs
                - Viewer demographics through rating analysis
                - Optimal content duration insights
                
                **Technologies Used**: Python, Pandas, Matplotlib, Seaborn
                """)
                
                st.markdown("""
                **Links**:  
                [<img src='https://img.icons8.com/ios-glyphs/30/4CAF50/github.png' width='20'/> GitHub](https://github.com/Kushank97/Netflix-Content-Analysis) | 
                [<img src='https://img.icons8.com/ios-glyphs/30/4CAF50/linkedin.png' width='20'/> LinkedIn Post](https://www.linkedin.com/feed/update/urn:li:activity:7337025775650357249/)
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
      
        with st.container():
            st.markdown('<div class="portfolio-project">', unsafe_allow_html=True)
            
            st.markdown('<h3 class="project-title">2. Ola Rides Data Analysis</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if os.path.exists("ola.png"):
                    st.image("ola.png", width=300, use_container_width=True, caption="Ola Rides Analysis Dashboard")
                else:
                    st.warning("Image not found: ola.png")
            
            with col2:
                st.markdown("""
                **Project Overview**:  
                Analysis of 100,000+ Ola ride records uncovering:
                - Customer behavior and ride patterns
                - Cancellation trends and reasons
                - Payment method preferences
                - Peak demand periods and pricing
                - Driver performance metrics
                
                **Technologies Used**: SQL, Power BI, Excel
                """)
                
                st.markdown("""
                **Links**:  
                [<img src='https://img.icons8.com/ios-glyphs/30/4CAF50/github.png' width='20'/> GitHub](https://github.com/Kushank97/-Ola-Rides-Data-Analysis-Project)
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
     
        with st.container():
            st.markdown('<div class="portfolio-project">', unsafe_allow_html=True)
            
            st.markdown('<h3 class="project-title">3. Music Store Data Analysis</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if os.path.exists("musicstore.png"):
                    st.image("musicstore.png", width=300, use_container_width=True, caption="Music Store Analysis")
                else:
                    st.warning("Image not found: musicstore.png")
            
            with col2:
                st.markdown("""
                **Project Overview**:  
                SQL and Python analysis of a music store database:
                - Sales trends and customer segmentation
                - Popular genres and artist performance
                - Inventory and purchasing patterns
                - Geographic sales distribution
                
                
                **Technologies Used**: PostgreSQL, Python, Pandas, Matplotlib
                """)
                
                st.markdown("""
                **Links**:  
                [<img src='https://img.icons8.com/ios-glyphs/30/4CAF50/github.png' width='20'/> GitHub](https://github.com/Kushank97/MusicStore_DataAnalysis) | 
                [<img src='https://img.icons8.com/ios-glyphs/30/4CAF50/linkedin.png' width='20'/> LinkedIn Post](https://www.linkedin.com/posts/kushank-sharma-72bb86296_dataanalysis-sql-python-activity-7332007039226863616-lda5)
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
       
        with st.container():
            st.markdown('<div class="portfolio-project">', unsafe_allow_html=True)
            
            st.markdown('<h3 class="project-title">4. E-Commerce Sales Dashboard</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if os.path.exists("dashboard.png"):
                    st.image("dashboard.png", width=300, use_container_width=True, caption="E-Commerce Dashboard")
                else:
                    st.warning("Image not found: dashboard.png")
            
            with col2:
                st.markdown("""
                **Project Overview**:  
                Interactive Power BI dashboard analyzing:
                - Sales performance by product/category
                - Customer demographics and behavior
                - Regional sales distribution
                - Seasonal trends and forecasting
                - Profitability analysis
                
                **Technologies Used**: Power BI, DAX, Power Query, Excel
                """)
                
                st.markdown("""
                **Links**:  
                [<img src='https://img.icons8.com/ios-glyphs/30/4CAF50/github.png' width='20'/> GitHub](https://github.com/Kushank97/E-Commerce-Sales-Analysis-Dashboard)
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

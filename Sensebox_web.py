import streamlit as st
import joblib
import pandas as pd
import os

# Set page config first
st.set_page_config(
    page_title="Sense Box AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models - use direct paths for Streamlit Cloud
try:
    spam_model = joblib.load("models/spam_classifier.pkl")
    language_model = joblib.load("models/lang_det.pkl")
    news_model = joblib.load("models/news_cat.pkl")
    review_model = joblib.load("models/review.pkl")
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    st.stop()

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
        }
        .model-info {
            background-color: rgba(40, 40, 80, 0.7) !important;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
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
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# Sidebar
with st.sidebar:
    try:
        st.image("images/kushank.png", caption="Sense Box AI", use_container_width=True)
    except:
        st.warning("Profile image not found")

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
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📞 Contact Us"):
        st.markdown("### Get in Touch")
        st.link_button("📧 Email", "mailto:kushanksharma97@gmail.com")
        st.link_button("💻 GitHub", "https://github.com/Kushank97")
        st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/kushank-sharma-72bb86296")

# Main content
st.markdown('<h1 class="title-text">🎯 Sense Box AI: Market Sentiment Engine</h1>', unsafe_allow_html=True)

# Tabs
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
            content = uploaded_file.getvalue().decode('utf-8')
            return pd.DataFrame({'text': content.split('\n')})
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# Tab 1: Spam Classifier
with tab1:
    st.header("📩 Spam Classifier")
    
    with st.container():
        st.markdown("""
        <div class="model-info">
        <h4>Model Description</h4>
        <p>This model classifies text messages as either spam or not spam using machine learning. 
        Trained on labeled data, it uses TF-IDF vectorization and algorithms like Logistic Regression 
        to detect promotional or harmful content.</p>
        </div>
        """, unsafe_allow_html=True)

    msg = st.text_input("Enter a message to classify")
    if st.button("Detect Spam"):
        pred = spam_model.predict([msg])
        if pred[0] == 0:
            st.success("❌ Spam Detected!")
            try:
                st.image("images/spams.webp", width=300)
            except:
                pass
        else:
            st.success("✅ Not Spam!")
            try:
                st.image("images/tick.jpg", width=300)
            except:
                pass

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

# Tab 2: Language Detection
with tab2:
    st.header("🌐 Language Detection")
    
    with st.container():
        st.markdown("""
        <div class="model-info">
        <h4>Model Description</h4>
        <p>This model identifies the language of any given text input. It analyzes character 
        and word patterns to distinguish between multiple languages.</p>
        </div>
        """, unsafe_allow_html=True)

    text = st.text_area("Enter text to detect language")
    if st.button("Detect Language"):
        pred = language_model.predict([text])
        st.success(f"🈯 Detected Language: **{pred[0]}**")

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

# Tab 3: Food Review Sentiment
with tab3:
    st.header("🍽️ Food Review Sentiment")
    
    with st.container():
        st.markdown("""
        <div class="model-info">
        <h4>Model Description</h4>
        <p>The sentiment analysis model determines whether a given text expresses positive or negative 
        emotions. It applies NLP techniques to understand tone and opinion.</p>
        </div>
        """, unsafe_allow_html=True)

    review = st.text_area("Enter a food review")
    if st.button("Analyze Sentiment"):
        pred = review_model.predict([review])
        if pred[0] == 0:
            st.success("👎 Negative Feedback")
        else:
            st.success("👍 Positive Feedback")

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

# Tab 4: News Classification
with tab4:
    st.header("📰 News Classification")
    
    with st.container():
        st.markdown("""
        <div class="model-info">
        <h4>Model Description</h4>
        <p>This model is currently under development to deliver more accurate and reliable category 
        predictions for news articles.</p>
        </div>
        """, unsafe_allow_html=True)

    try:
        st.image("images/under_construction.png", width=300)
    except:
        st.warning("Under construction image not found")

# Tab 5: Portfolio
with tab5:
    st.markdown('<h1 style="color:#4CAF50; text-align:center;">📊 Data Analyst Portfolio</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Project 1
    with st.container():
        st.markdown('<div class="portfolio-project">', unsafe_allow_html=True)
        st.markdown('<h3 class="project-title">1. Netflix Content Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            try:
                st.image("images/netflix.png", width=300, use_container_width=True, caption="Netflix Analysis Dashboard")
            except:
                st.warning("Netflix image not found")
        
        with col2:
            st.markdown("""
            **Project Overview**:  
            Comprehensive analysis of Netflix's content catalog revealing:
            - Content distribution (Movies vs TV Shows)
            - Release trends and seasonality patterns
            - Geographic production hubs
            
            **Technologies Used**: Python, Pandas, Matplotlib, Seaborn
            """)
            
            st.markdown("""
            **Links**:  
            [<img src='https://img.icons8.com/ios-glyphs/30/4CAF50/github.png' width='20'/> GitHub](https://github.com/Kushank97/Netflix-Content-Analysis)
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Project 2
    with st.container():
        st.markdown('<div class="portfolio-project">', unsafe_allow_html=True)
        st.markdown('<h3 class="project-title">2. Ola Rides Data Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            try:
                st.image("images/ola.png", width=300, use_container_width=True, caption="Ola Rides Analysis Dashboard")
            except:
                st.warning("Ola image not found")
        
        with col2:
            st.markdown("""
            **Project Overview**:  
            Analysis of 100,000+ Ola ride records uncovering:
            - Customer behavior and ride patterns
            - Cancellation trends and reasons
            - Payment method preferences
            
            **Technologies Used**: SQL, Power BI, Excel
            """)
            
            st.markdown("""
            **Links**:  
            [<img src='https://img.icons8.com/ios-glyphs/30/4CAF50/github.png' width='20'/> GitHub](https://github.com/Kushank97/-Ola-Rides-Data-Analysis-Project)
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
from joblib import load
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Initialize NLTK
nltk.download('stopwords')

# Set page config
st.set_page_config(
    page_title="Fake News Detection System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling and mobile responsiveness
st.markdown("""
<style>
    @import url('https://fonts.cdnfonts.com/css/cera-round-pro');

    html, body, [class*="css"]  {
        font-family: 'Cera Round Pro', sans-serif;
        background-color: #000000;
        color: white;
    }

    .header, .footer {
        text-align: center;
        padding: 2rem 0;
    }

    .title-circle {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: white;
        margin: 0 auto 1rem;
    }

    .hero {
        text-align: center;
        margin: 2rem 0;
    }

    .section-title {
        font-size: 2rem;
        margin: 1rem 0;
        text-align: center;
    }

    .button-container {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
        flex-wrap: wrap;
    }

    .stTextInput input, .stTextArea textarea {
        background-color: #111111;
        color: white;
        border: 1px solid #555;
    }
</style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model_components():
    try:
        model = load('logistic_regression_fakenews_model.pkl')
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None

model, vectorizer = load_model_components()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    return ' '.join(tokens)

# Word cloud generator
def generate_wordcloud(text):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap='viridis',
        max_words=100
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# --- UI SECTIONS ---

# Header
st.markdown("""
<div class="header">
    <div class="title-circle"></div>
    <h2>PROJECT: FAKE NEWS DETECTION SYSTEM (PROTOTYPE)</h2>
    <p>Landing Page - 2025</p>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <h1>FAKE NEWS DETECTOR</h1>
    <p>AI-Powered Content Analysis System</p>
</div>
""", unsafe_allow_html=True)

# News Input
st.markdown("""
<div class="section-title">News Analysis</div>
""", unsafe_allow_html=True)

input_text = st.text_area("Paste news article content here:", height=300)

col1, col2 = st.columns([1, 1])
with col1:
    analyze_btn = st.button("Analyze Content")
with col2:
    clear_btn = st.button("Clear Text")

if clear_btn:
    input_text = ""
    st.experimental_rerun()

# --- Analysis Logic ---
if analyze_btn and input_text.strip():
    with st.spinner("Analyzing content..."):
        try:
            clean_text = preprocess_text(input_text)
            text_vector = vectorizer.transform([clean_text])
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]

            st.markdown("---")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### Prediction")
                if prediction == 0:
                    st.error("🚨 Fake News Detected (Label: 0)")
                else:
                    st.success("✅ Real News (Label: 1)")

                st.progress(int(max(probabilities) * 100))
                st.metric("Confidence Score", f"{max(probabilities) * 100:.1f}%")

            with col_b:
                st.pyplot(generate_wordcloud(clean_text))

            with st.expander("Detailed Analysis"):
                tab1, tab2 = st.tabs(["Statistics", "Processed Text"])

                with tab1:
                    st.write("### Probability Distribution")
                    prob_df = pd.DataFrame({
                        "Label": ["Fake (0)", "Real (1)"],
                        "Probability": [probabilities[0], probabilities[1]]
                    })
                    st.bar_chart(prob_df.set_index("Label"))

                    col_x, col_y, col_z = st.columns(3)
                    with col_x:
                        st.metric("Original Length", f"{len(input_text):,} chars")
                    with col_y:
                        st.metric("Processed Length", f"{len(clean_text):,} chars")
                    with col_z:
                        st.metric("Unique Words", len(set(clean_text.split())))

                with tab2:
                    st.code(clean_text)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

elif analyze_btn and not input_text.strip():
    st.warning("Please enter some text to analyze")

# --- Future Work Section ---
st.markdown("""
<div class="section-title">💡 Future Work</div>
""", unsafe_allow_html=True)

with st.expander("Planned Improvements"):
    st.markdown("""
    - Improve generalization by training the model on more diverse and real-world news datasets.
    - Add multilingual support and language detection.
    - Explore BERT or transformer models for better accuracy.
    - Integrate live news feed scraping and streaming analysis.
    """)

# Footer
st.markdown("""
<div class="footer">
    <h3>THE END</h3>
    <p>Thanks for using our system!</p>
</div>
""", unsafe_allow_html=True)

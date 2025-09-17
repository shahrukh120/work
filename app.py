# app.py
import streamlit as st
import pickle
import string
import os
import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from PIL import Image
import pytesseract

# --- Page Configuration ---
st.set_page_config(page_title="SMS Spam Detector", page_icon="📧", layout="wide")

# --- NLTK and Model Setup ---
# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

ps = PorterStemmer()

# --- Helper Functions ---
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

@st.cache_resource
def load_model_and_vectorizer():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        return pytesseract.image_to_string(image)
    except Exception:
        return None

def save_feedback(message, actual_label):
    file_exists = os.path.isfile("feedback.csv")
    with open('feedback.csv', 'a') as f:
        if not file_exists:
            f.write("message,label\n")
        label_numeric = 1 if actual_label == "Spam" else 0
        cleaned_message = message.replace('\n', ' ').replace(',', ';')
        f.write(f'"{cleaned_message}",{label_numeric}\n')

# --- Load Model ---
tfidf, model = load_model_and_vectorizer()
if model is None or tfidf is None:
    st.error("❌ Model or Vectorizer files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the root directory.")
    st.stop()

# --- UI Layout ---
st.sidebar.title("ℹ️ About This App")
st.sidebar.info(
    "This **SMS Spam Detector** uses a trained Machine Learning model to classify "
    "messages as **Spam** or **Not Spam**.\n\n"
    "Navigate to the **Dashboard** to see feedback stats and to the **Admin** page to retrain the model."
)

st.markdown("<h1 style='text-align: center;'>✉️ SMS Spam Detector</h1>", unsafe_allow_html=True)

# Initialize session state for storing results
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'message' not in st.session_state:
    st.session_state.message = ""

# --- Tabs for Input ---
tab1, tab2 = st.tabs(["✍️ Text Input", "🖼️ Image Upload"])

with tab1:
    input_sms = st.text_area("Message", placeholder="Type or paste your SMS here...", height=150)
    if st.button("🔍 Check Text", use_container_width=True):
        if not input_sms.strip():
            st.warning("⚠️ Please enter a message to check.")
        else:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            prediction = model.predict(vector_input)[0]
            st.session_state.prediction_result = "Spam" if prediction == 1 else "Not Spam"
            st.session_state.message = input_sms

with tab2:
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if st.button("🖼️ Analyze Image", use_container_width=True):
        if uploaded_file is not None:
            extracted_text = extract_text_from_image(uploaded_file)
            if extracted_text and extracted_text.strip():
                st.info(f"**Extracted Text:**\n\n---\n\n{extracted_text}")
                transformed_sms = transform_text(extracted_text)
                vector_input = tfidf.transform([transformed_sms])
                prediction = model.predict(vector_input)[0]
                st.session_state.prediction_result = "Spam" if prediction == 1 else "Not Spam"
                st.session_state.message = extracted_text
            else:
                st.warning("⚠️ Could not extract meaningful text from the image.")
        else:
            st.warning("⚠️ Please upload an image first.")

# --- Display Prediction and Feedback (this is now outside the buttons) ---
if st.session_state.prediction_result:
    st.write("---")
    st.subheader("Prediction Result")
    if st.session_state.prediction_result == "Spam":
        st.error("🚨 This message is likely SPAM.")
    else:
        st.success("✅ This message seems SAFE.")

    st.write("---")
    st.subheader("Was this prediction correct?")
    
    col1, col2, col3 = st.columns([1,1,5]) # Make buttons smaller
    
    if col1.button("👍 Yes"):
        st.success("Thank you for your feedback!")
        st.session_state.prediction_result = None # Reset after feedback
    
    if col2.button("👎 No"):
        st.session_state.wrong_feedback = True

    if 'wrong_feedback' in st.session_state and st.session_state.wrong_feedback:
        st.warning("Please select the correct label:")
        
        r_col1, r_col2 = st.columns(2)
        
        if r_col1.button("It was actually SPAM"):
            save_feedback(st.session_state.message, "Spam")
            st.success("✅ Feedback saved! The model can now be improved.")
            st.session_state.wrong_feedback = False
            st.session_state.prediction_result = None # Reset
        
        if r_col2.button("It was actually NOT SPAM"):
            save_feedback(st.session_state.message, "Not Spam")
            st.success("✅ Feedback saved! The model can now be improved.")
            st.session_state.wrong_feedback = False
            st.session_state.prediction_result = None # Reset







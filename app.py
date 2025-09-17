import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from PIL import Image
import pytesseract

# -----------------
# ⚠️ IMPORTANT: If you're on Windows, you might need to set the Tesseract path manually.
# Example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# -----------------

# Download NLTK data (only needs to be done once)
nltk.download('stopwords')
nltk.download('punkt')

# Initialize Porter Stemmer
ps = PorterStemmer()

# --- Text Transformation Function ---
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    return " ".join(y)

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError:
        st.error("❌ Model or Vectorizer file not found. Please make sure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"⚠️ An error occurred while loading the model files: {e}")
        return None, None

tfidf, model = load_model_and_vectorizer()

# --- OCR Function ---
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
        return None

# --- Streamlit Page Config ---
st.set_page_config(page_title="SMS Spam Detector", page_icon="✉️", layout="wide")

# --- Sidebar Info ---
st.sidebar.title("ℹ️ About This App")
st.sidebar.info(
    "This **SMS Spam Detector** uses a trained Machine Learning model to classify "
    "messages as **Spam** or **Not Spam**.\n\n"
    "👉 You can either type/paste a message or upload an image of a message."
)

# --- Main Header ---
st.markdown("<h1 style='text-align: center;'>✉️ SMS Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Check if a message is Spam or Safe using AI</p>", unsafe_allow_html=True)
st.write("---")

# --- Tabs for Input ---
tab1, tab2 = st.tabs(["✍️ Text Input", "🖼️ Image Upload"])

# --- Tab 1: Text Input ---
with tab1:
    st.subheader("Enter a message below")
    input_sms = st.text_area("Message", placeholder="Type or paste your SMS here...")

    if st.button("🔍 Check Text", use_container_width=True):
        if model and tfidf:
            if not input_sms.strip():
                st.warning("⚠️ Please enter a message to check.")
            else:
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]

                if result == 1:
                    st.error("🚨 Spam Detected!")
                else:
                    st.success("✅ This message is Safe (Not Spam).")

# --- Tab 2: Image Upload ---
with tab2:
    st.subheader("Upload an image of a message")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if st.button("🖼️ Analyze Image", use_container_width=True):
        if model and tfidf:
            if uploaded_file is not None:
                extracted_text = extract_text_from_image(uploaded_file)
                st.info(f"**Extracted Text:**\n\n---\n\n{extracted_text}\n\n---")

                if extracted_text and extracted_text.strip():
                    transformed_sms = transform_text(extracted_text)
                    vector_input = tfidf.transform([transformed_sms])
                    result = model.predict(vector_input)[0]

                    if result == 1:
                        st.error("🚨 Spam Detected!")
                    else:
                        st.success("✅ This message is Safe (Not Spam).")
                else:
                    st.warning("⚠️ Could not extract meaningful text from the image.")
            else:
                st.warning("⚠️ Please upload an image first.")


  







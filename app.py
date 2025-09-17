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
# Block for 'stopwords'
try:
    nltk.data.find('corpora/stopwords')
except LookupError: # <- The fix
    nltk.download('stopwords')

# Block for 'punkt'
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # <- The fix
    nltk.download('punkt')

# Initialize Porter Stemmer
ps = PorterStemmer()

# --- Text Transformation Function ---
# This function should be the *exact* same one you used to preprocess text
# when you trained your model.
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# --- Load Model and Vectorizer ---
# Use st.cache_resource to load them only once
@st.cache_resource
def load_model_and_vectorizer():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError:
        st.error("Model or Vectorizer file not found. Please make sure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model files: {e}")
        return None, None

tfidf, model = load_model_and_vectorizer()


# --- OCR Function ---
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# --- Streamlit App Interface ---
st.title("✉️ SMS Spam Detector")
st.write("Enter a message or upload an image of a message to check if it's spam or not.")

# Create tabs for different input types
tab1, tab2 = st.tabs(["✍️ Text Input", "🖼️ Image Upload"])

# --- Tab 1: Text Input ---
with tab1:
    input_sms = st.text_area("Enter the message")
    if st.button("Check Text"):
        if model is None or tfidf is None:
            st.stop()
            
        if not input_sms.strip():
            st.warning("Please enter a message to check.")
        else:
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. Predict
            result = model.predict(vector_input)[0]
            # 4. Display
            if result == 1:
                st.header("🚨 Result: Spam")
            else:
                st.header("✅ Result: Not Spam")

# --- Tab 2: Image Upload ---
with tab2:
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if st.button("Check Image"):
        if model is None or tfidf is None:
            st.stop()

        if uploaded_file is not None:
            # Extract text from image
            extracted_text = extract_text_from_image(uploaded_file)
            st.info(f"**Extracted Text:**\n\n---\n\n{extracted_text}\n\n---")
            
            if extracted_text and extracted_text.strip():
                # 1. Preprocess
                transformed_sms = transform_text(extracted_text)
                # 2. Vectorize
                vector_input = tfidf.transform([transformed_sms])
                # 3. Predict
                result = model.predict(vector_input)[0]
                # 4. Display
                if result == 1:
                    st.header("🚨 Result: Spam")
                else:
                    st.header("✅ Result: Not Spam")
            else:
                st.warning("Could not extract any text from the image, or the extracted text is empty.")
        else:
            st.warning("Please upload an image first.")




  







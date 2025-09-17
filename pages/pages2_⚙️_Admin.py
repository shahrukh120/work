# pages/2_⚙️_Admin.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
from app import transform_text # Import the function from your main app

st.set_page_config(page_title="Model Admin", page_icon="⚙️", layout="wide")
st.title("⚙️ Model Management & Retraining")

def retrain_model():
    feedback_file = 'feedback.csv'
    original_data_file = 'spam.csv'
    
    if not os.path.exists(feedback_file):
        return "⚠️ No feedback data available to retrain the model."
    
    if not os.path.exists(original_data_file):
        return "❌ Original `spam.csv` not found. Cannot retrain."

    # 1. Load data
    original_data = pd.read_csv(original_data_file, encoding='ISO-8859-1')
    feedback_data = pd.read_csv(feedback_file)
    
    # 2. Prepare and combine datasets
    original_data = original_data[['v2', 'v1']]
    original_data.columns = ['message', 'label']
    original_data['label'] = original_data['label'].apply(lambda x: 1 if x == 'spam' else 0)
    
    combined_data = pd.concat([original_data, feedback_data], ignore_index=True).dropna()

    if combined_data.empty:
        return "⚠️ Combined dataset is empty after processing."

    # 3. Re-run training process
    st.write(f"Retraining model with {len(combined_data)} total samples ({len(original_data)} original + {len(feedback_data)} new).")
    
    combined_data['transformed_text'] = combined_data['message'].apply(transform_text)
    
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(combined_data['transformed_text']).toarray()
    y = combined_data['label'].values
    
    model = MultinomialNB()
    model.fit(X, y)
    
    # 4. Save the new model and vectorizer
    with open('vectorizer.pkl', 'wb') as f_vec:
        pickle.dump(tfidf, f_vec)
    with open('model.pkl', 'wb') as f_mod:
        pickle.dump(model, f_mod)
        
    return "✅ Model retrained successfully with new feedback!"

# --- UI for Retraining ---
st.subheader("Retrain the Model")
st.warning("⚠️ **Warning:** This will overwrite the current live model with a new one trained on the original data plus all collected user feedback.", icon="🤖")

if st.button("🔄 Start Retraining"):
    with st.spinner("Retraining in progress... This may take a moment."):
        status = retrain_model()
    st.success(status)
    # Clear cache to force Streamlit to reload the new model files
    st.cache_resource.clear()
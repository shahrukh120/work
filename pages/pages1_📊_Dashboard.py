# pages/1_📊_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Project Dashboard")

feedback_file = 'feedback.csv'

if os.path.exists(feedback_file):
    try:
        feedback_df = pd.read_csv(feedback_file, names=['message', 'label'], header=0)
        if feedback_df.empty:
            st.warning("Feedback file is empty. No data to display.")
        else:
            # --- Key Metrics ---
            total_feedback = len(feedback_df)
            st.metric(label="Total Feedback Entries Collected", value=total_feedback)
            
            # --- Visualizations ---
            st.subheader("Distribution of New Labels from Feedback")
            label_counts = feedback_df['label'].map({0: 'Not Spam', 1: 'Spam'}).value_counts()
            
            fig = px.pie(values=label_counts.values, 
                         names=label_counts.index, 
                         title="Feedback Label Distribution",
                         color=label_counts.index,
                         color_discrete_map={'Spam':'#EF553B', 'Not Spam':'#636EFA'})
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Raw Data ---
            st.subheader("Recent Feedback Data")
            st.dataframe(feedback_df.tail(10))
            
    except pd.errors.EmptyDataError:
        st.warning("Feedback file is empty. No data to display.")
    except Exception as e:
        st.error(f"An error occurred while reading the feedback file: {e}")
else:
    st.warning("No feedback data collected yet. Use the main app page to submit feedback!")
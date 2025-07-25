"""
DataInsight AI - Interactive Web Application

This module uses Streamlit to create a user-friendly web interface for the
DataInsight AI platform. It allows users to upload data, configure the
preprocessing task, run the automated pipeline, and download the results.
"""

import streamlit as st
import pandas as pd
import io
import joblib

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from automator import WorkflowOrchestrator
from config import settings

st.set_page_config(
    page_title="DataInsight AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(uploaded_file):
    """Caches the data loading to prevent reloading on every interaction."""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def show_lineage_report(roles: dict):
    """Displays the column roles and processing decisions."""
    st.subheader("📋 Processing Lineage Report")
    st.write("The automator has classified your columns into the following roles:")
    for role, columns in roles.items():
        if columns:
            with st.expander(f"**{role.replace('_', ' ').title()}** ({len(columns)} columns)"):
                st.write(columns)


st.title("🤖 DataInsight AI")
st.markdown("Your intelligent partner for automated data preprocessing and feature engineering.")

with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    st.markdown("---")
    
    # Initialize session state for dataframe
    if 'df' not in st.session_state:
        st.session_state.df = None

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.success("File loaded successfully!")
    
    if st.session_state.df is not None:
        st.header("2. Configure Your Task")
        
        task = st.selectbox(
            "Select your ML task:",
            options=['classification', 'regression', 'clustering'],
            help="Choose the goal for your machine learning model."
        )
        
        target_column = None
        if task in ['classification', 'regression']:
            # Filter out non-numeric targets for regression
            potential_targets = st.session_state.df.columns.tolist()
            if task == 'regression':
                potential_targets = st.session_state.df.select_dtypes(include='number').columns.tolist()
            
            target_column = st.selectbox(
                "Select the target column (label):",
                options=potential_targets,
                help="This is the column your model will try to predict."
            )
        
        st.session_state.task = task
        st.session_state.target_column = target_column

if st.session_state.df is None:
    st.info("Please upload a dataset using the sidebar to get started.")
else:
    st.header("Data Preview")
    st.dataframe(st.session_state.df.head())
    
    if 'task' in st.session_state:
        if st.button("🚀 Run Automated Preprocessing", type="primary", use_container_width=True):
            with st.spinner("The AI is analyzing your data and building the pipeline..."):
                try:
                    # Instantiate and run the orchestrator
                    orchestrator = WorkflowOrchestrator(
                        df=st.session_state.df,
                        target_column=st.session_state.target_column,
                        task=st.session_state.task
                    )
                    pipeline, roles = orchestrator.build()
                    
                    # Store results in session state
                    st.session_state.pipeline = pipeline
                    st.session_state.roles = roles
                    
                    X = st.session_state.df.drop(columns=[st.session_state.target_column]) if st.session_state.target_column else st.session_state.df
                    if st.session_state.task in ['classification', 'regression']:
                        y = st.session_state.df[st.session_state.target_column]
                        X_transformed = pipeline.fit_transform(X, y)
                    else: # Clustering
                        y = None
                        X_transformed = pipeline.fit_transform(X)

                    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                    st.session_state.processed_df = pd.DataFrame(X_transformed, columns=feature_names)
                    
                    if y is not None:
                        st.session_state.processed_df[st.session_state.target_column] = y.values

                    st.success("Preprocessing complete!")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    st.exception(e) # Provides a full traceback for debugging

if 'processed_df' in st.session_state:
    st.markdown("---")
    st.header("✨ Results")
    
    tab1, tab2, tab3 = st.tabs(["Processed Data", "Lineage Report", "Downloads"])

    with tab1:
        st.subheader("Preview of Processed Data")
        st.dataframe(st.session_state.processed_df.head())
        st.metric("Shape of Processed Data", f"{st.session_state.processed_df.shape[0]} rows, {st.session_state.processed_df.shape[1]} columns")

    with tab2:
        show_lineage_report(st.session_state.roles)

    with tab3:
        st.subheader("Download Artifacts")
        
        # 1. Download Processed Data
        csv = st.session_state.processed_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Processed Data (.csv)",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv",
        )
        
        # 2. Download Pipeline Object
        buffer = io.BytesIO()
        joblib.dump(st.session_state.pipeline, buffer)
        buffer.seek(0)
        st.download_button(
            label="📥 Download Pipeline Object (.joblib)",
            data=buffer,
            file_name="pipeline.joblib",
            mime="application/octet-stream",
            help="Use this file to apply the same transformations to new data."
        )
import streamlit as st
import pandas as pd
import numpy as np
import csv
import json
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from sklearn.inspection import permutation_importance
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
from utils import prepare_dataset,detect_problem_type, load_csv
from fi import get_feature_importance
from prompts import SYSTEM_PROMPT
from llm import build_llm_payload, generate_report, validate_report
from schemas import FeatureInsight, BusinessReport
from ml import train_model


# APP CONFIG

st.set_page_config(
    page_title="Feature Importance Analyzer",
    layout="wide"
)

st.title("📊 Feature Importance Analyzer App")

# LOAD DATA

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# MAIN FLOW

if uploaded_file is not None:

    df = load_csv(uploaded_file)

    st.success("File loaded successfully")
    st.dataframe(df.head())

    # TARGET SELECTION

    target_col = st.selectbox("Select target column", df.columns)

    df, X, y, missing_rows = prepare_dataset(df, target_col)

    if missing_rows > 0:
        st.warning(f"Missing values in target detected: {missing_rows} rows have been removed.")

    # PROBLEM TYPE DETECTION
    
    problem_type = detect_problem_type(y)

    st.info(f"Detected problem type: {problem_type}")

    # MODEL SELECTION

    problem_type = st.radio(
        "Problem type",
        ["Classification", "Regression"],
        index=0 if problem_type == "Classification" else 1
    )

    # MODEL TRAINING
    
    exp = (
    ClassificationExperiment() 
    if problem_type == "Classification" 
    else RegressionExperiment()
    )
   
    train_clicked = st.button("Train and Analyze")
    
    if train_clicked:
           
        with st.spinner("Training and evaluating machine learning models"):

            results = train_model(
                 exp=exp,
                 df=df,
                 target_col=target_col,
            )

        
        st.session_state["training_results"] = results
        
       
    if "training_results" not in st.session_state:
        st.warning("Please train the model first.")
        st.stop()
   
    # LOAD RESULTS FROM SESSION STATE

    training_results = st.session_state["training_results"]

    best_model = training_results["best_model"]
    leaderboard = training_results["leaderboard"]
    best_metrics = training_results["best_metrics"]
    X = training_results["X"]
    y = training_results["y"]

    st.write("Best model:", type(best_model).__name__)
    st.dataframe(leaderboard.head(5))

       
    # FEATURE IMPORTANCE PIPELINE

    fi_dict = get_feature_importance(best_model, X, y)

    # UI TABLE

    if len(fi_dict["features"]) != len(fi_dict["importance"]):
        st.error(
        f"Mismatch: features={len(fi_dict['features'])}, "
        f"importance={len(fi_dict['importance'])}"
    )
        st.write(fi_dict)
        st.stop()

    fi_df = pd.DataFrame({
        "feature": fi_dict["features"],
        "importance": fi_dict["importance"]
    }).sort_values("importance", ascending=False)

    st.subheader("Feature Importance (Table)")
    st.dataframe(fi_df)

  
# LLM REPORT - OPTIONAL 

    st.divider()

    st.subheader("AI Business Report")

    st.info("""AI report generation requires a valid OpenAI API key. 
            The key is used only for the current session and is not stored.""")
    
    generate_ai = st.checkbox("Generate AI business report")

    user_api_key = None

    if generate_ai:
        user_api_key = st.text_input("Enter your OpenAI API key", type="password")
    
    if st.button("Generate Report"):
        if not user_api_key:
            st.error("Please enter your OpenAI key.")
            st.stop()

        payload = build_llm_payload(
                target_col = target_col,
                problem_type = problem_type,
                best_model = best_model,
                best_metrics = best_metrics,
                fi_dict = fi_dict,
                fi_df = fi_df
            )

        with st.spinner("Generating AI business report..."):

                try:
                    report = generate_report(
                            payload=payload,
                            api_key=user_api_key)
                    
                    validate_report(report)

                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    st.stop()

        # DISPLAY REPORT IN UI 

        st.subheader("Executive Summary")
        st.write(report.executive_summary)

        st.subheader("Key Drivers")

        for f in report.key_drivers:
                st.markdown(f"""
            **{f.feature}**
            - Importance: {f.importance}
            - Interpretation: {f.interpretation}
            - Business impact: {f.business_implication}
            - Confidence: {f.confidence_note}
            """)
                
        st.subheader("Recommendations")
        for rec in report.recommendations:
                st.markdown(f"- {rec}")
            
        st.subheader("Quick Wins")
        for win in report.quick_wins:
                st.markdown(f"- {win}") 
            

        st.subheader("Risks")
        for risk in report.risks:
                st.markdown(f"- {risk}")
              
        st.subheader("Final Summary")
        st.write(report.final_summary)
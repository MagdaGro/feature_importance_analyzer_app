# 📊 Feature Importance Analyzer

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://twoja-aplikacja.onrender.com)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![PyCaret](https://img.shields.io/badge/PyCaret-3.x-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4.1--mini-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)


An interactive Streamlit application for automatically training machine learning models, identifying the most important predictive features, and generating AI-powered business reports.

The application is designed for non-technical users who want to understand which variables drive model predictions and how these insights can support business decision-making.

---

##  Features

* Upload any CSV dataset
* Automatic delimiter detection
* Automatic problem type detection (Classification / Regression)
* Automatic machine learning model selection using PyCaret
* Feature importance analysis
* Permutation importance fallback for unsupported models
* Interactive results table
* AI-generated business report powered by OpenAI
* Structured LLM output validated with Pydantic
* Retry logic for robust AI report generation
* No API key storage – users provide their own OpenAI API key when generating reports

---

##  Technologies

* Python
* Streamlit
* PyCaret
* Scikit-learn
* Pandas
* NumPy
* OpenAI API
* Pydantic

---

## Project Structure

```text
app.py              # Streamlit application
ml.py               # Machine learning pipeline
llm.py              # AI report generation
utils.py            # Helper functions
schemas.py          # Pydantic schemas
prompts.py          # System prompt for LLM
requirements.txt
README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/feature-importance-analyzer.git

cd feature-importance-analyzer
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
```

Activate it:

Windows

```bash
.venv\Scripts\activate
```

macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## How It Works

1. Upload a clean, machine-learning-ready CSV dataset.
2. Select the target variable.
3. The application detects whether the task is classification or regression.
4. PyCaret automatically trains and compares multiple machine learning models.
5. The best-performing model is selected.
6. Feature importance is calculated.
7. Optionally generate an AI-powered business report using your own OpenAI API key.

---

## AI Business Report

The application can generate a structured business report including:

* Executive Summary
* Key Business Drivers
* Business Interpretation of Features
* Recommendations
* Quick Wins
* Risks and Limitations
* Final Summary

The report is generated using the OpenAI API.

Your API key:

* is entered manually,
* is used only during the current session,
* is never stored by the application.

---

## Notes

* Works with both classification and regression datasets.
* Rows containing missing values in the target column are automatically removed.
* Feature importance method depends on the selected model.
* Business insights are generated only from the provided model outputs.

---

## License

This project is provided for educational and portfolio purposes.

---

## Author

Created by Magda as part of a machine learning and AI portfolio project.

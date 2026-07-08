feature-importance-analyzer/
│
├── app.py              # Streamlit UI
├── utils.py            # Uploading files, preprocessing, helpers
├── ml.py               # Models training PyCaret
├── fi.py               # Feature importance
├── llm.py              # AI Report generating
├── prompts.py          # System prompt
├── schemas.py          # Pydantic schemas
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│    ├── diagram.excalidraw.png
│    ├── structure.md
│    └── screenshot.png
│
└── sample_data/
    ├── iris.csv
    └── titanic.csv
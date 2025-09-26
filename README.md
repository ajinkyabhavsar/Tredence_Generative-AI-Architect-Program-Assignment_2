 
# Personalized Course Recommendation Engine

This project implements a **semantic course recommender** using embeddings, FAISS vector search, and a simple **Streamlit UI**.
Learners can enter their background, completed courses, and interests to receive the **top-5 most relevant course suggestions**.

---

## Features
- Converts course descriptions into **embeddings** with [SentenceTransformers]
- Uses **FAISS** for efficient similarity search (cosine similarity).
- Excludes already completed courses from results.
- Interactive **Streamlit web app** for easy use.
- Extensible for larger datasets and alternative embedding providers (Google, OpenAI, etc.).

---

## Project Structure
.
├── app.py # Main Streamlit app
├── Dataset.csv # Sample dataset (auto-created if missing)
├── requirements.txt # Python dependencies
└── README.md # Project documentation



---

## Installation

1. Clone the repository or copy the files.

2. Install dependencies:
bash
pip install -r requirements.txt

3. Usage
Run the Streamlit app:
bash
streamlit run app.py

Open the link shown in the terminal (usually http://localhost:8501) in your browser.

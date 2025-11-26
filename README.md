# Assessli Demo Project  
A simple demo engine that analyzes behavioral statements and turns them into actionable insights through:

- Sentence embeddings  
- Dimensionality reduction (UMAP)  
- Adaptive-K clustering (smart KMeans-based)  
- Rule-based persona generation  
- Interactive visualization in Streamlit  

This is a **mini behavior-intelligence engine** that demonstrates how Assessli could build deeper, structured, explainable insight systems.

---

## Why This Exists
This repository is a **demo** submitted alongside an application to Assessli.  
The goal:  
Show initiative, technical creativity, and the ability to build intelligence systems that could evolve into future Assessli features.

This project is **fully local**, requires **no LLMs**, and demonstrates:

- Natural language understanding  
- Pattern extraction  
- Visualization  
- Personality clustering  
- User-friendly summarization  

---

## Folder Structure

assessli/
│
├── engine/
│ ├── embeddings.py # Converts sentences → numeric vectors
│ ├── reducer.py # UMAP reduction → 2D points
│ ├── clusterer.py # Adaptive KMeans clustering (auto-selects K)
│ └── persona_generator.py # Rule-based persona summaries
│
├── streamlit_app/
│ └── app.py # Main UI for uploading & visualizing
│
├── data/
│ ├── sample_statements.txt # Optional test file
│ ├── reduced_points.csv # Saved output
│ └── clusters.csv #future
│
├── notebooks/
│ └── analysis_demo.ipynb # Small notebook for experiments #future
│
├── models/
│ └── (empty for now) # Future: custom finetuned embedding models
│
├── requirements.txt
└── README.md


---

## Installation  
You need Python **3.10 or 3.11**.

```bash
pip install -r requirements.txt

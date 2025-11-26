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

V0.3 — Planned Technical Enhancements
1. Hybrid Embedding Stack (Local + Cloud Models)

v0.3 will introduce a hybrid embedding architecture:

Local fast embeddings (MiniLM or E5-Small)

Optional high-quality cloud embeddings (OpenAI text-embedding-3-large / Voyage-Large / Cohere Embed v3)

Automatic similarity benchmarking to choose best embedding per dataset

Outcome: Higher semantic resolution, better cluster separation, better personality persona accuracy.

2. Advanced Dimensionality Reduction Pipeline

Switch from a single-stage UMAP → to a two-stage hybrid reduction:

Stage 1: PCA (retain 95% variance)

Stage 2: UMAP / t-SNE for human-interpretable 2D projection

This will improve cluster separability and reduce noise in downstream clustering.

3. New Clustering Engine (Beyond KMeans)

v0.3 will introduce:

a) Hybrid Clustering System

KMeans for global structure

DBSCAN / HDBSCAN-like logic for local density-based clusters

Cluster merging using cosine centroid similarity

b) Automated Cluster Optimization

Auto-search best k using silhouette, Davies-Bouldin, Calinski-Harabasz

L2-normalized cluster centroids

Automatic outlier detection

Outcome: More reliable persona groups, less noise, more interpretability.

4. Data-Driven Persona Generation (Semi-LLM but Controlled)

Instead of rule-based text templates, v0.3 will have:

A trait extraction engine using phrase clustering

Automatic extraction of:

Cognitive style markers

Motivation words

Behavioral signals

Optional local small LLM (e.g., Llama 3–8B locally with quantization) for:

Persona naming

Trait summarization

Cluster-level descriptions

No cloud LLMs required unless configured.
5. Insight Graphs & Behavioral Maps

New UI components:
Cluster-level heatmaps

Trait network graphs

Persona similarity map

Cross-person correlation visualization (if multi-user analysis enabled)

Built using:

Plotly

NetworkX

Interactive Streamlit components

6. Modular Pipeline Architecture

v0.3 will introduce a fully modular engine:

pipeline/
│
├── embeddings/
│   ├── miniLM.py
│   ├── e5.py
│   └── openai_embedder.py
│
├── reducers/
│   ├── pca.py
│   └── umap.py
│
├── clustering/
│   ├── kmeans.py
│   ├── dbscan.py
│   └── hybrid_clusterer.py
│
└── personas/
    ├── keyword_trait_extractor.py
    ├── cluster_profiler.py
    └── persona_generator.py
Each module can be swapped by config.
This demonstrates strong software architecture maturity.

7. Cloud Deployable (Vercel + Streamlit + API Mode)

While v0.2 is fully local, v0.3 will enable:

Vercel deployment

API endpoints for:

/embed

/reduce

/cluster

/persona

Making Assessli’s behavior-intelligence engine usable as a micro-service.



---

## Installation  
You need Python **3.10 or 3.11**.

```bash
pip install -r requirements.txt

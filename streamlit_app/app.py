# app.py
# Streamlit front-end for the Assessli behavior-map engine (v0.2, KMeans version)

import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO

# -----------------------------
# Engine imports
# -----------------------------
try:
    from ..engine.embeddings import EmbeddingEngine
    from ..engine.reducer import UMAPReducer
    from ..engine.clusterer import ClusterEngine
    from ..engine.persona_generator import PersonaGenerator
except Exception:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from engine.embeddings import EmbeddingEngine
    from engine.reducer import UMAPReducer
    from engine.clusterer import ClusterEngine
    from engine.persona_generator import PersonaGenerator

import plotly.express as px

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="Assessli — Behavior Map (v0.2 KMeans)",
    layout="wide"
)
st.title("Assessli — Behavior Map (v0.2, KMeans Version)")
st.caption("Pipeline: SentenceEmbeddings (MiniLM) → UMAP → KMeans → Persona cards")

# Optional header image
DEMO_HEADER_IMAGE = "/mnt/data/1000086665.jpg"
if os.path.exists(DEMO_HEADER_IMAGE):
    st.image(DEMO_HEADER_IMAGE, use_column_width=True)

# Layout
col1, col2 = st.columns([1, 2])

# -----------------------------
# LEFT PANEL – INPUT + SETTINGS
# -----------------------------
with col1:
    st.header("Input")
    st.markdown(
        "Paste **one short statement per line** describing your behavior, motivation, or learning patterns.\n(10–30 statements recommended)"
    )

    text_input = st.text_area(
        "Statements (one per line)",
        height=240,
        placeholder=(
            "Example:\n"
            "I learn best by building small projects.\n"
            "I get demotivated by long lectures.\n"
            "I prefer pair programming.\n"
        ),
    )

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Or upload CSV with column `statement`", type=["csv"]
    )
    st.markdown("---")

    st.header("Engine Settings (v0.2 KMeans)")
    n_neighbors = st.slider("UMAP: n_neighbors", 5, 50, 15)
    min_dist = st.slider("UMAP: min_dist", 0.0, 0.99, 0.1, step=0.05)
    num_clusters = st.slider(
        "KMeans: Number of clusters", min_value=2, max_value=10, value=3
    )

    run_button = st.button("Generate Behavior Map")

# -----------------------------
# Utility
# -----------------------------


def load_statements(text_input, uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "statement" not in df.columns:
                st.error("CSV must contain a 'statement' column.")
                return []
            return df["statement"].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return []
    else:
        return [l.strip() for l in text_input.splitlines() if l.strip()]


# -----------------------------
# RIGHT PANEL – OUTPUT PLACEHOLDER
# -----------------------------
with col2:
    st.header("Preview / Outputs")
    st.info("Run the engine to see results here.")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if run_button:
    statements = load_statements(text_input, uploaded_file)

    if not statements or len(statements) < 3:
        st.error("Please enter at least 3 statements.")
    else:
        # Embeddings
        with st.spinner("Loading embedding model (MiniLM)..."):
            embedder = EmbeddingEngine()
            embedder.fit(statements)

            embeddings = embedder.encode(statements)

        # UMAP reduction
        reducer = UMAPReducer(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2
        )

        # Convert embeddings to dense if sparse to avoid eigsh error
        if hasattr(embeddings, "toarray"):
            embeddings = embeddings.toarray()

        coords = reducer.fit_transform(embeddings)

        # KMeans clustering
        clusterer = ClusterEngine()
        # pass clusters as positional arg
        labels = clusterer.fit(coords)
        clusters = clusterer.get_cluster_dict(statements, labels)

        # Personas
        persona_gen = PersonaGenerator()
        personas = persona_gen.generate_personas(clusters)

        # Output dataframe
        df_out = pd.DataFrame({
            "statement": statements,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": labels
        })

        # -----------------------------
        # Visualization
        # -----------------------------
        fig = px.scatter(
            df_out,
            x="x",
            y="y",
            color=df_out["cluster"].astype(str),
            hover_data=["statement", "cluster"],
            title="Behavior Map — UMAP (MiniLM) + KMeans",
            width=1000,
            height=600
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=0.5)))

        with col2:
            st.subheader("Behavior Map (KMeans)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Clusters & Personas")
            for cid, pdata in personas.items():
                st.markdown(f"### Cluster {cid}: **{pdata['title']}**")
                st.write(pdata["description"])
                if pdata.get("keywords"):
                    st.write("**Keywords:**", ", ".join(pdata["keywords"]))
                samples = clusters.get(cid, [])[:4]
                if samples:
                    st.write("**Sample statements:**")
                    for s in samples:
                        st.write("•", s)
                st.markdown("---")

# -----------------------------
# Downloads
# -----------------------------
st.subheader("Downloads")

# CSV download (safe)
st.download_button(
    "Download CSV",
    df_out.to_csv(index=False).encode("utf-8"),
    "assessli_behavior_map.csv",
    "text/csv",
)

# -----------------------------
# Safe PNG export
# -----------------------------
png_data = None
try:
    # Try generating PNG using Plotly -> requires Kaleido
    png_data = fig.to_image(format="png")
except Exception as e:
    st.warning(
        "PNG export is unavailable in this environment (Kaleido missing). "
        "You can still screenshot the map above."
    )

# PNG download button only if successful
if png_data:
    st.download_button(
        "Download Map (PNG)",
        png_data,
        "assessli_behavior_map.png",
        "image/png",
    )
# -----------------------------
# Safe local save (always works)
# -----------------------------
out_dir = "/tmp/assessli_v02_outputs"
os.makedirs(out_dir, exist_ok=True)

# Save CSV always
csv_path = os.path.join(out_dir, "behavior_map.csv")
df_out.to_csv(csv_path, index=False)

# Save PNG only if available
if png_data:
    png_path = os.path.join(out_dir, "behavior_map.png")
    with open(png_path, "wb") as f:
        f.write(png_data)

st.success(f"Saved files under `{out_dir}` (temporary storage)")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "### Notes\n"
    "- Version uses **UMAP + KMeans** for clustering.\n"
    "- MiniLM embeddings run fully local.\n"
    "- v0.3 will add persona PDF export + better UI theme."
)

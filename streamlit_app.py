import streamlit as st
import nomic
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Text Embedding Visualizer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model Caching ---
# Cache the sentence transformer model to avoid reloading it on every run
@st.cache_resource
def get_model():
    """Loads and caches the SentenceTransformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

model = get_model()

# --- Default Text ---
DEFAULT_TEXT = """
Gemini is a family of generative AI models.
The sun is the star at the center of the Solar System.
Paris is the capital and most populous city of France.
Machine learning is a field of study in artificial intelligence.
The Earth revolves around the sun.
London is the capital of the United Kingdom.
Large language models are trained on massive amounts of text data.
The moon is Earth's only natural satellite.
Berlin is the capital of Germany.
"""

# --- Streamlit UI ---

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    st.write("Enter your Nomic API key and the text you want to visualize.")
    
    # Get Nomic API key from user
    nomic_api_key = st.text_input(
        "Nomic API Key", 
        type="password",
        help="Get your free API key from https://atlas.nomic.ai/"
    )

    st.header("About")
    st.info(
        "This app uses a sentence-transformer model to generate embeddings for each line of text. "
        "It then uses Nomic Atlas to create an interactive 2D map of those embeddings."
    )

# --- Main Page ---
st.title("🗺️ Interactive Text Embedding Visualizer")
st.write(
    "Paste your text below (one sentence or document per line). The app will generate embeddings "
    "and visualize them in a 2D map where similar sentences appear closer together."
)

# Text area for user input
text_input = st.text_area("Enter Text (one sentence per line)", DEFAULT_TEXT, height=250)

# Button to trigger visualization
if st.button("Generate and Visualize Embeddings", type="primary"):
    if not nomic_api_key:
        st.error("🚨 Please enter your Nomic API key in the sidebar.")
    elif not text_input.strip():
        st.warning("📋 Please enter some text to visualize.")
    else:
        # Set the API key
        try:
            nomic.login(nomic_api_key)
        except Exception as e:
            st.error(f"Failed to login to Nomic. Please check your API key. Error: {e}")
            st.stop()
            
        # 1. Process Text
        with st.spinner("Processing text..."):
            lines = [line.strip() for line in text_input.split('\n') if line.strip()]
            
            if len(lines) < 2:
                st.warning("Please provide at least two lines of text to create a map.")
                st.stop()

        # 2. Generate Embeddings
        with st.spinner(f"Generating embeddings for {len(lines)} sentences... This may take a moment."):
            embeddings = model.encode(lines, show_progress_bar=False)
            
            # Create metadata for Nomic
            data = [{'text': line, 'id': i} for i, line in enumerate(lines)]

        # 3. Create Nomic Map
        with st.spinner("Creating your interactive Nomic map..."):
            try:
                project = nomic.map_embeddings(
                    embeddings=np.array(embeddings),
                    data=data,
                    id_field='id',
                    colorable_fields=['text'],
                    map_name="Streamlit Text Embedding Visualization",
                    map_description="A map of sentence embeddings generated via a Streamlit app."
                )
                
                st.success("🎉 Map created successfully!")
                
                # Display the map using an iframe
                st.components.v1.iframe(project.maps[0].map_link, height=800)

            except Exception as e:
                st.error(f"An error occurred while creating the Nomic map: {e}")

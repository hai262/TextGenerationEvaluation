import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import itertools
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ----------------------------
# Monkey Patch for Fraction Error (Python 3.11+)
import fractions
import nltk.translate.bleu_score as bleu
class PatchedFraction(fractions.Fraction):
    def __new__(cls, numerator, denominator, _normalize=True):
        return super().__new__(cls, numerator, denominator)
bleu.Fraction = PatchedFraction

nltk.download('punkt')

# ----------------------------
# Page Configuration
st.set_page_config(page_title="Text Generation Evaluation with BLEU Scores", layout="wide")

# ----------------------------
# Custom CSS for an Attractive, Professional Look
# 1) Remove padding/margin from the main container
# 2) Style the cover text
st.markdown(
    """
    <style>
    /* Remove default padding/margin so cover image can fill the screen */
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
    }
    .stApp {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header-custom {
        font-size: 3.0em;
        font-weight: 600;
        color: #1920f7; /* Tomato */
        text-align: justify;
        margin: 0 20px;
    }
    .description-custom {
        font-size: 1.8em;
        color: #db3d12; /* DodgerBlue */
        text-align: justify;
        margin: 0 20px;
    }
    .sub-header {
        font-size: 1.6em;
        color: #34495e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Sidebar: Organized into Clear Sections
with st.sidebar:
    st.header("Input & Settings")
    
    # --- Reference Text Section ---
    with st.expander("Reference Text", expanded=True):
        ref_input_method = st.radio("Select Input Method", ("Upload File", "Paste Text"))
        if ref_input_method == "Upload File":
            uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
            if uploaded_file:
                reference_text = uploaded_file.read().decode("utf-8")
            else:
                reference_text = ""
        else:
            reference_text = st.text_area("Paste Reference Text", height=150)
    
    # --- Markov Chain Settings ---
    with st.expander("Markov Chain Settings", expanded=True):
        st.markdown("Adjust the parameters for the text generation model:")
        ngram_order = st.slider("n-gram Order", min_value=2, max_value=4, value=2, step=1)
        num_words_to_generate = st.slider("Words to Generate", 20, 200, 50, step=10)
        num_samples = st.slider("Number of Samples", 3, 20, 5, step=1)
    
    # --- Smoothing Method Selection ---
    with st.expander("Smoothing Method", expanded=True):
        smoothing_method = st.selectbox(
            "Select Smoothing Method", 
            options=["Method 1", "Method 2", "Method 3", 
                     "Method 4", "Method 5", "Method 6", "Method 7"],
            index=0
        )
    
    # --- Advanced Grid Search Settings ---
    with st.expander("Advanced Grid Search Settings", expanded=False):
        enable_grid_search = st.checkbox("Enable Grid Search", value=False)
        if enable_grid_search:
            grid_ngram = st.multiselect("n-gram Orders", options=[2, 3, 4], default=[2, 3, 4])
            grid_words = st.multiselect("Words to Generate", options=[20, 50, 100], default=[20, 50, 100])
            grid_samples = st.number_input("Samples for Grid Search", min_value=1, max_value=20, value=3, step=1)
            grid_smoothing = st.multiselect(
                "Smoothing Methods",
                options=["Method 1", "Method 2", "Method 3", 
                         "Method 4", "Method 5", "Method 6", "Method 7"],
                default=["Method 1"]
            )
        compare_all = st.checkbox("Compare All Smoothing Methods", value=False)

# ----------------------------
# Cover Page: Show Only if No Reference Text Provided
if not reference_text or reference_text.strip() == "":
    cover_html = """
    <div style="position: absolute; top:0; left:0; width:80vw; height:100vh; overflow:hidden; text-align:center; color:white;">
      <img src="https://cfcdn.decopy.ai/features/humanizer/ogimg.jpg"
           alt="Cover Image"
           style="width:100%; height:100%; object-fit:cover; opacity:0.8;">
      <div style="position: absolute; top: 30%; left: 50%; transform: translate(-50%, -30%); width: 90%;">
        <div class="main-header-custom">Text Generation Evaluation Using BLEU Scores</div>
        <div class="description-custom">
          Evaluate the quality of AI-generated text using BLEU scores by comparing it with a reference text.
          Adjust parameters and explore advanced grid search options to fine-tune your model.
        </div>
      </div>
    </div>
    """
    st.markdown(cover_html, unsafe_allow_html=True)
    st.stop()

# ----------------------------
# Helper Functions
def build_ngram_model(text, n):
    """Build an n-gram Markov Chain model from text."""
    text_clean = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text_clean.lower().split()
    model = {}
    for i in range(len(words) - n):
        key = tuple(words[i:i+n])
        next_word = words[i+n]
        model.setdefault(key, []).append(next_word)
    return model

def generate_text_from_model(model, start_tuple, num_words):
    """Generate text using the Markov Chain model."""
    current_tuple = start_tuple
    output = list(current_tuple)
    for _ in range(num_words - len(current_tuple)):
        next_words = model.get(current_tuple, None)
        if not next_words:
            break
        next_word = random.choice(next_words)
        output.append(next_word)
        current_tuple = tuple(output[-len(current_tuple):])
    return " ".join(output)

def calculate_bleu(reference, generated_texts, smoothing_method_str="method1"):
    """Calculate BLEU scores for generated texts using a specified smoothing method."""
    sf = SmoothingFunction()
    smoothing = getattr(sf, smoothing_method_str)
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    scores = []
    for text in generated_texts:
        candidate_tokens = nltk.word_tokenize(text.lower())
        score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)
        scores.append(score)
    return scores

def compare_smoothing_methods(reference, generated_texts):
    """Compare average BLEU scores for all smoothing methods."""
    methods = ["method1", "method2", "method3", "method4", "method5", "method6", "method7"]
    avg_scores = {}
    for m in methods:
        scores = calculate_bleu(reference, generated_texts, smoothing_method_str=m)
        avg_scores[m] = np.mean(scores)
    return avg_scores

def run_grid_search(reference, grid_ngram, grid_words, grid_samples, grid_smoothing):
    """Perform grid search over model parameters and smoothing methods."""
    results = []
    for n in grid_ngram:
        model = build_ngram_model(reference, n)
        unique_keys = list(model.keys())
        if not unique_keys:
            continue
        for w in grid_words:
            for s_method in grid_smoothing:
                generated_texts = [generate_text_from_model(model, random.choice(unique_keys), w)
                                   for _ in range(grid_samples)]
                method_key = s_method.lower().replace(" ", "")
                scores = calculate_bleu(reference, generated_texts, smoothing_method_str=method_key)
                avg_bleu = np.mean(scores)
                results.append({
                    "n-gram Order": n,
                    "Words Generated": w,
                    "Smoothing Method": s_method,
                    "Average BLEU": avg_bleu
                })
    return pd.DataFrame(results)

# ----------------------------
# Main Application Logic
if reference_text:
    # Build the main Markov model using provided n-gram order
    model_main = build_ngram_model(reference_text, ngram_order)
    unique_keys_main = list(model_main.keys())
    
    if not unique_keys_main:
        st.error("Reference text is too short for the chosen n-gram order. Please use a longer text.")
    else:
        # Generate samples and calculate BLEU scores for the main run
        machine_texts = [generate_text_from_model(model_main, random.choice(unique_keys_main), num_words_to_generate)
                         for _ in range(num_samples)]
        selected_method = smoothing_method.lower().replace(" ", "")
        bleu_scores = calculate_bleu(reference_text, machine_texts, smoothing_method_str=selected_method)
        
        # Organize main output into tabs
        main_tabs = st.tabs(["Overview", "Generated Samples", "Visualizations", "Parameter Tuning", "Download Results"])
        
        # --- Tab 1: Overview ---
        with main_tabs[0]:
            st.markdown("<div class='sub-header'>Overview & Settings</div>", unsafe_allow_html=True)
            st.text_area("Reference Text Preview (first 1000 characters)", reference_text[:1000], height=180)
            st.markdown(f"**Markov Chain Settings:** n-gram order = {ngram_order}, words = {num_words_to_generate}, samples = {num_samples}")
            st.markdown(f"**Smoothing Method:** {smoothing_method}")
        
        # --- Tab 2: Generated Samples ---
        with main_tabs[1]:
            st.markdown("<div class='sub-header'>Generated Samples & BLEU Scores</div>", unsafe_allow_html=True)
            for i, text in enumerate(machine_texts):
                st.markdown(f"**Sample {i+1} (BLEU Score: {bleu_scores[i]:.4f})**")
                st.write(text)
        
        # --- Tab 3: Visualizations ---
        with main_tabs[2]:
            st.markdown("<div class='sub-header'>BLEU Score Visualizations</div>", unsafe_allow_html=True)
            # Histogram
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(bleu_scores, bins=10, kde=True, color="blue", alpha=0.7)
            plt.axvline(np.mean(bleu_scores), color='red', linestyle='dashed', label=f'Avg: {np.mean(bleu_scores):.4f}')
            plt.title("BLEU Score Distribution")
            plt.xlabel("BLEU Score")
            plt.ylabel("Frequency")
            plt.legend()
            st.pyplot(fig)
            # Scatter Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.scatter(range(1, len(bleu_scores)+1), bleu_scores, color='purple', s=50)
            plt.axhline(np.mean(bleu_scores), color='red', linestyle='dashed', label=f'Avg: {np.mean(bleu_scores):.4f}')
            plt.title("BLEU Scores per Sample")
            plt.xlabel("Sample Number")
            plt.ylabel("BLEU Score")
            plt.legend()
            st.pyplot(fig)
            # Boxplot
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=bleu_scores, color="green")
            plt.title("Boxplot of BLEU Scores")
            st.pyplot(fig)
        
        # --- Tab 4: Parameter Tuning ---
        with main_tabs[3]:
            st.markdown("<div class='sub-header'>Parameter Tuning & Smoothing Comparison</div>", unsafe_allow_html=True)
            if compare_all:
                avg_scores = compare_smoothing_methods(reference_text, machine_texts)
                df_compare = pd.DataFrame(list(avg_scores.items()), columns=["Smoothing Method", "Avg BLEU Score"])
                st.dataframe(df_compare)
                st.markdown("**Bar Chart: Smoothing Method Comparison**")
                fig, ax = plt.subplots(figsize=(8,5))
                sns.barplot(data=df_compare, x="Smoothing Method", y="Avg BLEU Score", palette="viridis")
                plt.title("Average BLEU Score by Smoothing Method")
                plt.ylabel("Avg BLEU Score")
                st.pyplot(fig)
            else:
                st.info("Enable 'Compare All Smoothing Methods' in the sidebar to view comparisons.")
        
        # --- Tab 5: Download Results ---
        with main_tabs[4]:
            st.markdown("<div class='sub-header'>Download BLEU Scores</div>", unsafe_allow_html=True)
            df_bleu = pd.DataFrame({
                "Sample Number": range(1, num_samples + 1),
                "BLEU Score": bleu_scores
            })
            st.dataframe(df_bleu)
            csv_bleu = df_bleu.to_csv(index=False)
            st.download_button("Download BLEU Scores CSV", data=csv_bleu, file_name="bleu_scores.csv", mime="text/csv")
        
        # --- Optional Grid Search Tab ---
        if enable_grid_search:
            grid_tabs = st.tabs(["Grid Search Results", "Grid Search Visualizations"])
            with grid_tabs[0]:
                st.markdown("<div class='sub-header'>Grid Search Results</div>", unsafe_allow_html=True)
                df_grid = run_grid_search(reference_text, grid_ngram, grid_words, grid_samples, grid_smoothing)
                if df_grid.empty:
                    st.error("Grid search did not return any results. Adjust grid parameters or use a longer reference text.")
                else:
                    st.dataframe(df_grid)
            with grid_tabs[1]:
                st.markdown("<div class='sub-header'>Grid Search Heatmaps</div>", unsafe_allow_html=True)
                if not df_grid.empty:
                    for s_method in grid_smoothing:
                        df_subset = df_grid[df_grid["Smoothing Method"] == s_method]
                        if df_subset.empty:
                            continue
                        pivot = df_subset.pivot(index="n-gram Order", columns="Words Generated", values="Average BLEU")
                        st.markdown(f"**Smoothing Method: {s_method}**")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".4f")
                        plt.title(f"Heatmap ({s_method})")
                        st.pyplot(fig)
else:
    st.info("⚠️ Please upload or paste a reference text to begin analysis.")

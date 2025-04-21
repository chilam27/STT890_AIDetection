import streamlit as st

# Page configuration
st.title("Underlying Methodology")

# Tabs within this page
tab1, tab2 = st.tabs(["Summary", "Code Notebook"])

# Tab 1: Methodology Summary
with tab1:
    st.subheader("Methodology Used in This Project")
    st.markdown("""
    ### Dataset for training the model
    The dataset used for this project is the [AI Text Detection Pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile) by Artem Yatsenko, designed for AI text detection tasks and focused on long-form text and essays. It includes both human and AI-generated text, with sources for human text comprising Reddit Writing Prompts, OpenAI WebText, HC3 (Human Responses), and IvyPanda essays, while AI-generated text is derived from outputs of GPT-2, GPT-3 (Pairwise-Davinci and Synthetic-Instruct-Davinci-Pairwise), GPT-J (Synthetic-Instruct-GPT-J-Pairwise), and ChatGPT (from Twitter data, HC3 ChatGPT responses, ChatGPT prompts, and EmergentMind). The dataset consists of 1.39 million records, with 73.8% being human text and 26.2% AI-generated text.

    ### Text pre-processing steps
    Preprocessing steps were minimal due to the relatively clean text data, involving lowercasing, removal of special characters and extra whitespace, and TF-IDF feature extraction (limited to 5000 features, incorporating unigrams and bigrams, with English stopwords removed).

    ### Model Selection
    Five machine learning models were selected for the classification task: Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN). Logistic Regression served as a strong baseline with probabilistic outputs and interpretability, while Random Forest provided robust ensemble learning by combining decision trees to reduce overfitting and improve generalization. Gradient Boosting sequentially corrected prior errors for complex textual patterns, and SVM demonstrated effectiveness in high-dimensional spaces and sparse TF-IDF representations. KNN, as a non-parametric, instance-based learner, captured local patterns and performed well with balanced class distributions. These models collectively enabled a comprehensive evaluation across linear, ensemble-based, margin-based, and instance-based approaches to tackle this classification challenge.
    
    ### Deployment
    Implemented in a Streamlit app with processed and vectorized input and real-time prediction. 
    
    ### Streamlit App Pipeline Summary
    1. Text Preprocessing:
       - Lowercasing, removing special characters and extra whitespace
    2. TF-IDF Feature Extraction:
       - Max 5000 features, unigrams and bigrams, English stopwords removed
    3. Classify Text with Best Performing Model:
       - Random Foreset Classifier
    """)

# Tab 2: Render Notebook
with tab2:
    st.subheader("Project Code")
    
    # display notebook as html
    path_to_html = "TextPreprocessing_ModelTraining.html"
    
    with open(path_to_html,'r') as f: 
        html_data = f.read()
    
    # Show in webpage
    st.components.v1.html(html_data, scrolling = True, height = 800)

import streamlit as st
from rag_setup import retrieve_relevant_entries
from openai_integration import process_query
import pickle
from utils import create_accordion_html, check_password
import torch
from sentence_transformers import SentenceTransformer

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Load pre-initialized RAG objects
@st.cache_resource
def init_rag():
    try:
        with open('rag_df.pkl', 'rb') as f:
            df = pickle.load(f)
        with open('rag_embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        # Load the model
        device = torch.device("cpu")
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        model.load_state_dict(torch.load('rag_model.pkl', map_location=device))
        return df, embeddings, model
    except FileNotFoundError:
        st.error("RAG files not found. Please run create_rag_files.py first.")
        return None, None, None

df, embeddings, model = init_rag()

st.title("Zinc Fellows Finder")
st.subheader("Find Fellows based on their skills and background")

# SIDEBAR
st.sidebar.header("About")
st.sidebar.info(
    "So many amazing people in the Zinc Fellows list, so little time to look through them! "
    "Ask questions about their skills, background, or interests, and get AI-powered responses. "
    "***Fellows database last updated on 14 June***"
)

st.sidebar.caption(
    "Do you want something added or changed here? Message Rus"
)

# MAIN BIT
question_input = st.text_input(label="Ask your question", placeholder="Ask any question about the Fellows database. E.g. 'Who would know about how to sell to the NHS?'", label_visibility="hidden")

# Button to send question
if st.button("Ask question"):
    question_words = question_input.split()
    if len(question_words) >= 3:
        with st.spinner("Processing your question..."):
            try:
                # Retrieve relevant entries using the user-defined threshold
                relevant_entries = retrieve_relevant_entries(question_input, df, embeddings, model)
                
                if len(relevant_entries) > 0:
                    # Process query and show OpenAI response
                    response = process_query(question_input, relevant_entries)
                    st.success("Here's what I found:")
                    st.markdown(f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; font-size: 18px;'>{response}</div>", unsafe_allow_html=True)
                    
                    # Display relevant fellows
                    st.divider()
                    st.caption("Possibly Relevant Fellows:")
                    accordion_html = create_accordion_html(relevant_entries)
                    st.components.v1.html(accordion_html, height=400, scrolling=True)
                else:
                    st.warning("No closely matching fellows found. Try lowering the relevance threshold or rephrasing your question.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please ask a longer question (at least 3 words).")
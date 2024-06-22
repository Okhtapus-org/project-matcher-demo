import streamlit as st
import pandas as pd
from rag_setup import initialize_rag, retrieve_relevant_entries
from openai_integration import process_query

# Initialize RAG system
@st.cache_resource
def init_rag():
    return initialize_rag()

df, index, model = init_rag()

st.title("Zinc Fellows Finder")
st.header("Find Fellows based on their skills and background")

# SIDEBAR
st.sidebar.header("About")
st.sidebar.info(
    "There are a lot of Zinc Fellows in the database and ain't nobody's got time to read it all."
    "To save time, ask questions about their skills, background, or interests, and get AI-powered responses."
    " ***Fellows database last updated on 14 June***"
)

threshold = st.sidebar.slider(
    "Relevance Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.6,  # default value
    step=0.05,
    help="Adjust this to control how closely fellows need to match your query. Higher values mean stricter matching."
)

# MAIN BIT
question_input = st.text_input(label="Ask your question", placeholder="Ask any question about the Fellows database", label_visibility="hidden")

# Button to send question
if st.button("Ask question"):
    question_words = question_input.split()
    if len(question_words) >= 3:
        with st.spinner("Processing your question..."):
            try:
                # Retrieve relevant entries using the user-defined threshold
                relevant_entries, similarities = retrieve_relevant_entries(question_input, df, index, model, threshold=threshold)
                
                if len(relevant_entries) > 0:
                    # Process query with OpenAI
                    response = process_query(question_input, relevant_entries)
                    
                    # Display the AI-generated response
                    st.success("Here's what I found:")
                    st.markdown(f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; font-size: 18px;'>{response}</div>", unsafe_allow_html=True)
                    
                    # Display relevant fellows
                    st.subheader(f"Relevant Fellows (Threshold: {threshold:.2f}):")
                    for (_, fellow), similarity in zip(relevant_entries.iterrows(), similarities):
                        st.write(f"**{fellow['Name']}** - {fellow['Role Title']} (Relevance: {similarity:.2f})")
                        with st.expander("See more"):
                            st.write(f"**Bio:** {fellow['Bio']}")
                            st.write(f"**Wants to engage by:** {fellow['Wants to engage by']}")
                            st.write(f"**VB Priority area(s):** {fellow['VB Priority area(s)']}")
                            st.write(f"**Sector/Type:** {fellow['Sector/ Type']}")
                            st.write(f"**Spike:** {fellow['Spike']}")
                else:
                    st.warning("No closely matching fellows found. Try lowering the relevance threshold or rephrasing your question.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please ask a longer question (at least 3 words).")
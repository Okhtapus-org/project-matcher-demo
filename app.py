import streamlit as st
from rag_setup import retrieve_relevant_entries
from openai_integration import process_query
import pickle
from utils import create_accordion_html, check_password
import torch
from sentence_transformers import SentenceTransformer

# if not check_password():
#     st.stop()  # Do not continue if check_password is not True.


# Load pre-initialized RAG objects
@st.cache_resource
def init_rag():
    try:
        with open("rag_df_1.pkl", "rb") as f:
            df = pickle.load(f)
        with open("rag_embeddings_1.pkl", "rb") as f:
            embeddings = pickle.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.load_state_dict(torch.load("rag_model_1.pkl", map_location=device))
        model.to(device)

        return df, embeddings, model
    except FileNotFoundError:
        st.error("RAG files not found. Get someone to run create_rag_files.py first.")
        return None, None, None


df, embeddings, model = init_rag()

# Set up custom CSS for white background, black text, and Quicksand font
st.markdown(
    """
    <style>
        /* Set main background color */
        .stApp {
            background-color: white;
        }

        /* Set all text color to black */
        .css-10trblm, .css-1v3fvcr, .css-18e3th9, .css-1sbz9lr, .css-1aumxhk, .css-145kmo2 {
            color: black !important;
        }

        /* Title and subheader specific styling */
        h1, h3, .css-1v3fvcr {
            color: black !important;
            font-family: 'Quicksand', sans-serif;
        }
         /* Title and subheader specific styling */
        h1, h2 {
            color: black !important;
            font-family: 'Quicksand', sans-serif;
        }

        /* Title font and styling */
        .css-10trblm {
            font-family: 'Quicksand', sans-serif;
            font-size: 2em;
        }

        /* Center the logo in the sidebar */
        [data-testid="stSidebar"] > div:first-child {
            display: flex;
            justify-content: center;
            padding: 20px 0;
        }

        /* Customize button style */
        .stButton button {
            background-color: #3772FF; 
            color: white;
            border-radius: 10px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #d2ff27;
            color: #000000;
        }
        /* Add a link to the Google Fonts stylesheet for Quicksand */
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;700&display=swap');
    </style>
""",
    unsafe_allow_html=True,
)

# Add the logo and title
st.image("logo-white.png", width=100)
st.title("Okhtapus Project Finder")
st.subheader("Find projects based on their relevance")

# Sidebar information
st.sidebar.header("About")
st.sidebar.info(
    """
    At Okhtapus, enablers such as government bodies, corporations, and conservation organizations 
    connect with solution innovators and funders to advance impactful ocean projects and site development 
    where they hold the key levers to advancing such projects and the support structures around them.
    """
)

# st.sidebar.caption(
#     "Do you want something added or changed here? Message Rus"
#     "  \nFellows database last updated on 30 Oct 2024"
#     "  \nDB link: https://airtable.com/appDtdgZ1tvVW7rhv/shrmBQwT87j1xH3Nh/tbleA8k4UFnl16Tsk"
# )

# with st.sidebar.expander("A note on privacy"):
#     st.write(
#         """
#         General data bit

#         As you've seen, the page is password protected.
#         You're querying the contents of a publicly available (but not publicly shared) Airtable database owned by Zinc.
#         The emails and LinkedIn URLs have been removed. Names have been partially anonimised by only
#         keeping the first letter of the surname.

#         "AI" bit

#         The relevant profile finding happens locally (i.e. privately) and only the partially anonimised contents of the relevant
#         profiles are seen by a remote LLM.
#         The actual answer comes from Open AI's gpt-4o-mini. Only what is visible to you as the user of this app is
#         sent to OpenAI, this *should* all be public info.
#         OpenAI have been asked not to train their models on the inputs.
#     """
#     )

# MAIN BIT
question_input = st.text_input(
    label="Ask your question",
    placeholder="Ask any question about the Projects database. E.g. 'Can you help me find projects on coral reef restoration?'",
    label_visibility="hidden",
)

# Button to send question
if st.button("Ask question"):
    question_words = question_input.split()
    if len(question_words) >= 3:
        with st.spinner("Processing your question..."):
            try:
                # Retrieve relevant entries using the user-defined threshold
                relevant_entries = retrieve_relevant_entries(
                    question_input, df, embeddings, model
                )

                if len(relevant_entries) > 0:
                    # Process query and show OpenAI response
                    response = process_query(question_input, relevant_entries)
                    # st.success("Here's what I found:")
                    # st.markdown(
                    #     f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; font-size: 18px;'>{response}</div>",
                    #     unsafe_allow_html=True,
                    # )

                    # Display relevant fellows
                    st.divider()
                    st.caption("Possibly Relevant Projects:")
                    accordion_html = create_accordion_html(relevant_entries)
                    st.components.v1.html(accordion_html, height=400, scrolling=True)
                else:
                    st.warning(
                        "No closely matching fellows found. Try lowering the relevance threshold or rephrasing your question."
                    )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please ask a longer question (at least 3 words).")

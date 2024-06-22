import streamlit as st
import pandas as pd
import re

st.title("Zinc Fellows Finder")

st.header("Find a Fellow who's a good match for you")
name_input = st.text_input(label="Enter your name", placeholder="Enter your name and we will find some fellows for you", label_visibility="hidden")

# Button to check name
if st.button("Find your match"):
    # Regular expression to match only letters, apostrophes, m-dashes, n-dashes, and spaces
    if re.match(r"^[a-zA-Z' \u2013\u2014]+$", name_input):
        # Check if the input contains at least two words
        words = name_input.split()
        if len(words) >= 2 and all(len(word) >= 2 for word in words):
            try:
                # Try to read the CSV file
                df = pd.read_csv('vb8founder.csv')
                # Search for the name in the CSV file
                profile = df[df['Name'].str.fullmatch(name_input, case=False, na=False)]
                if not profile.empty:
                    # Save the profile to a Python object
                    profile_data = profile.to_dict(orient='records')[0]
                    st.success("Profile found")
                    # TODO: Add the AI profile matching
                else:
                    st.error("Founder name not found, make sure to spell it exactly as in the gallery")
            except FileNotFoundError:
                st.error("CSV file not found. Make sure vb8founder.csv is in the same folder as app.py")
        else:
            st.error("Name must contain at least two words, each with at least two letters.")
    else:
        st.error("Oops that doesn't look like a valid name.")

st.header("Find Fellows based on their skills and background")
question_input = st.text_input(label="Ask your question", placeholder="Ask any question about the Fellows database", label_visibility="hidden")

# Button to send question
if st.button("Ask question"):
    question_words = question_input.split()
    if len(question_words) >= 3:
        st.success("Nice question")
        # TODO: Add the AI question and answer
    else:
        st.error("Ask a longer question.")
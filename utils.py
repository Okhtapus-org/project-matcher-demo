import pandas as pd
import streamlit as st
import hmac

def create_accordion_html(relevant_entries):
    accordion_html = """
    <style>
    .accordion {
      background-color: #f1f1f1;
      color: #444;
      cursor: pointer;
      padding: 18px;
      width: 100%;
      text-align: left;
      border: none;
      outline: none;
      transition: 0.4s;
      margin-bottom: 5px;
    }
    .active, .accordion:hover {
      background-color: #ccc;
    }
    .panel {
      padding: 0 18px;
      background-color: white;
      display: none;
      overflow: hidden;
    }
    </style>

    <script>
    function toggleAccordion(accordionId) {
      var panel = document.getElementById(accordionId);
      if (panel.style.display === "block") {
        panel.style.display = "none";
      } else {
        panel.style.display = "block";
      }
    }
    </script>
    """

    for i, (_, fellow) in enumerate(relevant_entries.iterrows()):
        accordion_html += f"""
        <button class="accordion" onclick="toggleAccordion('panel{i}')">
            {fellow['Name']} - {fellow['Role Title']}
        </button>
        <div id="panel{i}" class="panel">
        """
        
        fields_to_display = [
            ("Bio", "Bio"),
            ("Wants to engage by", "Wants to engage by"),
            ("VB Priority area(s)", "VB Priority area(s)"),
            ("Sector/Type", "Sector/ Type"),
            ("Spike", "Spike")
        ]
        
        for display_name, field_name in fields_to_display:
            if pd.notna(fellow[field_name]) and fellow[field_name] != "" and fellow[field_name].lower() not in ["n/a", "none"]:
                accordion_html += f"<p><strong>{display_name}:</strong> {fellow[field_name]}</p>"
        
        accordion_html += "</div>"

    return accordion_html

# *** PASSWORD CHECK ***
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False
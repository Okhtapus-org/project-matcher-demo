import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_prompt(query, relevant_entries):
    """
    Generate a prompt for the OpenAI API based on the query and relevant entries.
    """
    prompt = f"""You are an AI assistant helping to find information about Ocean Projects based on their region and project goals. 
    Use the following information to answer the query: "{query}"

    Relevant Fellow Information:
    """

    for _, entry in relevant_entries.iterrows():
        prompt += f"""
        Enabler Name: {entry['Enabler Name']}
        Time Period/Date of Completion: {entry['Time Period/Date of Completion']}
        Summary: {entry['Summary']}
        Regions: {entry['Regions']}
        Sector(s): {entry['Sector(s)']}
        10 Ocean Decade Challenges: {entry['10 Ocean Decade Challenges']}
        Project: {entry['Project']}
        
        """

    prompt += (
        "\nBased on this information, please provide a concise answer to the query."
    )

    return prompt


def query_openai(prompt, model="gpt-4o-mini"):
    """
    Send a query to the OpenAI API and return the response.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides information about Zinc Fellows.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,  # Adjust as needed
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"


def process_query(query, relevant_entries):
    """
    Process a query using the OpenAI API.
    """
    prompt = generate_prompt(query, relevant_entries)
    return query_openai(prompt)


# Test the OpenAI integration
if __name__ == "__main__":
    # This is just a mock-up for testing. In the real app, you'll get this from rag_setup.py
    mock_entries = pd.DataFrame(
        {
            "Name": ["John Doe"],
            "Role Title": ["Tech Entrepreneur"],
            "Bio": ["Experienced in AI and machine learning"],
            "Wants to engage by": ["Mentoring"],
            "VB Priority area(s)": ["Technology"],
            "Sector/ Type": ["Tech"],
            "Spike": ["AI"],
            "Hoping to gain by getting involved with Zinc": ["Networking"],
        }
    )

    test_query = "Who has experience in AI?"
    result = process_query(test_query, mock_entries)
    print(f"Query: {test_query}")
    print(f"Response: {result}")

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
    prompt = f"""You are an AI assistant helping to find information about Ocean Projects based on their region and project goals. 
    Use the following information to answer the query: "{query}"

    Relevant Fellow Information:
    """

    for _, entry in relevant_entries.iterrows():
        prompt += f"""
        Enabler Organization Name: {entry['Enabler Organization Name']}
        Role: {entry['Applicant Role/ Position']}
        Summary: {entry['Summary']}
        Regions: {entry['Regions']}
        Project Summary: {entry['summary of Project/Site?']}
        Organization type: {entry['Organization type']}
        10 Ocean Decade Challenges: {entry['10 Ocean Decade Challenges']}
        Project/Site?: {entry['Project/Site?']}
        
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
            "Enabler Organization Name": ["Enabler Organization Name"],
            "Role": ["Applicant Role/ Position"],
            "Summary": ["Summary"],
            "Regions": ["Regions"],
            "Project Summary": ["summary of Project/Site?"],
            "Organization type": ["Organization type"],
            "10 Ocean Decade Challenges": ["10 Ocean Decade Challenges"],
            "Project/Site?": ["Project/Site?"],
        }
    )

    test_query = "Who has experience in AI?"
    result = process_query(test_query, mock_entries)
    print(f"Query: {test_query}")
    print(f"Response: {result}")

import os
from anthropic import Anthropic

def extract_job_urls(file_name):
    # Initialize the Anthropic client
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Read the content from the file
    with open(file_name, 'r') as file:
        content = file.read()

    # Define the prompt
    prompt = "hello, claude"

    # Send the prompt to the Claude model using the Anthropic library
    try:
        response = client.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="claude-3-haiku-20240307",
        )

        # Check if the response is received
        if response:
            # Print the raw response
            print("Response received:")
            print(response)

            # Extract the message content from the response
            response_data = response.json()
            message_content = response_data['messages'][0]['content']

            # Print the message content
            print("Message content:")
            print(message_content)
            return message_content

        else:
            print("No response received from the API")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
import pickle
from rag_setup import initialize_rag
from tqdm import tqdm
import time

def run_with_progress_bar(total=100):
    for i in tqdm(range(total), desc="Initializing RAG"):
        time.sleep(0.1)  # Simulate work being done
        yield

try:
    print("Starting RAG initialization...")
    
    # Initialize RAG with progress bar
    progress = run_with_progress_bar()
    df, index, model = initialize_rag(progress)
    next(progress)  # Ensure progress bar reaches 100%

    print("RAG initialization complete. Saving objects...")

    # Save the objects
    with open('rag_df.pkl', 'wb') as f:
        pickle.dump(df, f)
    print("Saved rag_df.pkl")

    with open('rag_index.pkl', 'wb') as f:
        pickle.dump(index, f)
    print("Saved rag_index.pkl")

    with open('rag_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved rag_model.pkl")

    print("RAG objects saved successfully.")
    print("Initialization and saving process completed successfully!")

except Exception as e:
    print(f"An error occurred during the initialization or saving process: {str(e)}")
    print("Initialization failed. Please check the error message above and try again.")
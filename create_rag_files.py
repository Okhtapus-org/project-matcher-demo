import pickle
import os
import sys
import csv
import torch
from tqdm import tqdm
import time
from rag_setup import initialize_rag

# Pseudonimises file, then sets up vector files for RAG. Run this every time you update the DB file
# Usage: python create_rag_files.py vb8_fellows.csv

def anonymize_csv(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        
        fieldnames = [field for field in reader.fieldnames if field not in ['Website', 'Email Address']]
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        name_column = next((field for field in fieldnames if 'name' in field.lower()), None)
        
        if not name_column:
            print("Warning: No 'Name' column found. Proceeding without name anonymization.")
        
        for row in reader:
            row.pop('Website', None)
            row.pop('Email Address', None)
            
            if name_column:
                name_parts = row[name_column].split()
                if len(name_parts) > 1:
                    row[name_column] = f"{' '.join(name_parts[:-1])} {name_parts[-1][0]}"
            
            writer.writerow(row)

    print(f"Anonymized file has been created: {output_file}")

def run_with_progress_bar(total=100):
    for i in tqdm(range(total), desc="Processing"):
        time.sleep(0.1)
        yield

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_rag_files.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = f"{os.path.splitext(input_file)[0]}_anonymized.csv"

    try:
        print(f"Starting anonymization process for {input_file}...")
        anonymize_csv(input_file, output_file)

        print("Starting RAG initialization...")
        progress = run_with_progress_bar()
        df, embeddings, model = initialize_rag(output_file)
        next(progress)
        print("RAG initialization complete. Saving objects...")

        with open('rag_df_1.pkl', 'wb') as f:
            pickle.dump(df, f)
        print("Saved rag_df_1.pkl")

        with open('rag_embeddings_1.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
        print("Saved rag_embeddings_1.pkl")

        # Save the model state dict
        torch.save(model.state_dict(), 'rag_model_1.pkl')
        print("Saved rag_model_1.pkl")

        print("RAG objects saved successfully.")
        print("Initialization and saving process completed successfully!")

    except Exception as e:
        print(f"An error occurred during the process: {str(e)}")
        print("Process failed. Please check the error message above and try again.")

if __name__ == "__main__":
    main()
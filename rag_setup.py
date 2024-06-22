import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['combined_text'] = df.apply(lambda row: f"{row['Name']} {row['Role Title']} {row['Bio']} {row['Wants to engage by']} {row['VB Priority area(s)']} {row['Sector/ Type']} {row['Spike']} {row['Hoping to gain by getting involved with Zinc']}", axis=1)
    return df

def create_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['combined_text'].tolist())
    return embeddings, model

def setup_vector_store(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def retrieve_relevant_entries(query, df, index, model, top_k=5, threshold=0):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector.astype('float32'), top_k)
    
    # Convert distances to similarities (assuming cosine distance)
    similarities = 1 - distances / 2.0
    
    # Filter based on threshold
    mask = similarities[0] >= threshold
    filtered_indices = indices[0][mask]
    filtered_similarities = similarities[0][mask]
    
    # Sort by similarity (highest first)
    sorted_indices = filtered_similarities.argsort()[::-1]
    
    relevant_entries = df.iloc[filtered_indices[sorted_indices]]
    return relevant_entries, filtered_similarities[sorted_indices]

def initialize_rag():
    file_path = 'vb8_fellows.csv'
    df = load_and_preprocess_data(file_path)
    embeddings, model = create_embeddings(df)
    index = setup_vector_store(embeddings)
    return df, index, model

# Remove or comment out the following lines:
# test_query = "Find fellows interested in children's health"
# relevant_entries = retrieve_relevant_entries(test_query, df, index, model)
# print(relevant_entries)
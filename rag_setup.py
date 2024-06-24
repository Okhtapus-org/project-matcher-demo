import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['combined_text'] = df.apply(lambda row: f"{row['Name']} {row['Role Title']} {row['Bio']} {row['Wants to engage by']} {row['VB Priority area(s)']} {row['Sector/ Type']} {row['Spike']} {row['Hoping to gain by getting involved with Zinc']}", axis=1)
    return df

def create_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['combined_text'].tolist())
    return embeddings, model

def retrieve_relevant_entries(query, df, embeddings, model, top_k=5):
    query_embedding = model.encode([query])
    similarities = np.dot(embeddings, query_embedding.T).squeeze()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

def initialize_rag():
    file_path = 'vb8_fellows.csv'
    df = load_and_preprocess_data(file_path)
    embeddings, model = create_embeddings(df)
    return df, embeddings, model
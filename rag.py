from dotenv import load_dotenv
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

knowledge_base = [
    "Quantum computing uses qubits instead of bits for processing.",
    "Artificial Intelligence is revolutionizing various industries.",
    "Machine Learning involves training models on data to make predictions.",
    "FAISS is an efficient similarity search library used for retrieval tasks.",
    "Neural networks consist of multiple layers that learn complex patterns.",
]

model = SentenceTransformer("all-MiniLM-L6-v2") 
embeddings = np.array(model.encode(knowledge_base))


index = faiss.IndexFlatL2(embeddings.shape[1]) 
index.add(embeddings) 



def retrieve_relevant_text(query, top_k=1):
    query_embedding = np.array(model.encode([query]))
    distances, indices = index.search(query_embedding, top_k)
    retrieved_texts = [knowledge_base[i] for i in indices[0]]
    return "\n".join(retrieved_texts)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  
client = OpenAI(api_key=openai_api_key)

def generate_response(query):
    retrieved_text = retrieve_relevant_text(query)
    prompt = f"Context: {retrieved_text}\nUser Query: {query}\nAnswer:"
    response = client.completions.create(
        model="gpt-3.5-turbo",  
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Step 6: Test our simple RAG system
if __name__ == "__main__":
    query = "What is quantum computing?"
    retrieved_text = retrieve_relevant_text(query)
    response = generate_response(query)
    
    print("\n**Retrieved Text:**")
    print(retrieved_text)
    print("\n**RAG Response:**")
    print(response)
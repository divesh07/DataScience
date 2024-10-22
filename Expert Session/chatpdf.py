import streamlit as st
import os
import numpy as np
import pandas as pd
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import LatexTextSplitter
from sentence_transformers import SentenceTransformer
import cohere

# Initialize Sentence Transformer model
model_name = "paraphrase-MiniLM-L6-v2"  # Adjust model as needed
sentence_transformer_model = SentenceTransformer(model_name)

# Load and process the PDF file
pdf_path = 'Kavita Kunder Assign 2.pdf'
loader = PyPDFLoader(pdf_path)
documents = loader.load()

embeddings = []
documents_text = []
sources = []

latex_splitter = LatexTextSplitter(chunk_size=1000, chunk_overlap=50)

for docu in documents:
    docs = latex_splitter.create_documents([docu.page_content])
    for document in docs:
        document_embedding = sentence_transformer_model.encode(document.page_content)
        embeddings.append(document_embedding)
        documents_text.append(document.page_content)
        sources.append(docu.metadata.get('source', 'Unknown'))

# Create FAISS index
embedding_dimension = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dimension)
index.add(np.array(embeddings, dtype='float32'))

# Save index and document details
if not os.path.exists("Vector_Store"):
    os.makedirs("Vector_Store")
df = pd.DataFrame({'documents': documents_text, 'source': sources})
df.to_csv('Vector_Store/docs.csv', index=False)
faiss.write_index(index, 'Vector_Store/vector_db.index')

# Streamlit app layout
st.title("Document Search and Q&A")

query = st.text_input("Enter your query:")

if query:
    query_embedding = sentence_transformer_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, k=3)
    
    threshold = 50  # Define your threshold for distance match
    if distances[0][0] > threshold:
        st.write("Please ask a relevant question.")
    else:
        combined_similar_documents_content = []
        similar_documents_sources = []
        for i in indices[0]:
            similar_document_content = df.loc[i, 'documents']
            combined_similar_documents_content.append(similar_document_content)
            print(i)
            similar_document_source = df.loc[i, 'source']
            similar_documents_sources.append(similar_document_source)
        
        combined_similar_documents_content = ' '.join(combined_similar_documents_content)

        print(combined_similar_documents_content)
        print(list(set(similar_documents_sources)))
        cohere_prompt = f"Based on the document page {combined_similar_documents_content}, answer the question in a comical manner: '{query}'"

        # Call Cohere API for response generation
        co = cohere.Client('<Your api key>')  # Replace with your actual API key
        cohere_response = co.generate(
            model='command',
            prompt=cohere_prompt,
            max_tokens=4000,
            temperature=0.9,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )

        # Display Cohere API response
        st.write("Bot Response:")
        st.write(cohere_response.generations[0].text)
        st.write("Sources:")
        st.write(list(set(similar_documents_sources)))

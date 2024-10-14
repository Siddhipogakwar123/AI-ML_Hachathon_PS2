import json
from transformers import AutoTokenizer, AutoModel
import torch

# Load the dataset
with open('/content/corpus.json', 'r') as f:
    corpus = json.load(f)



# Preprocess the corpus
corpus_docs = [doc['body'] for doc in corpus]

# Load the same model for both query and document encoding
retrieval_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
retrieval_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def encode_query(query):
    # Use the same model to encode the query
    inputs = retrieval_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = retrieval_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
import faiss
import numpy as np

# Load the Sentence-BERT or MiniLM model for dense retrieval
retrieval_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
retrieval_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# Load the Cross-Encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def embed_document(text):
    # Embed document using the same model (MiniLM)
    inputs = retrieval_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = retrieval_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Embed all documents in the corpus
document_embeddings = np.vstack([embed_document(doc) for doc in corpus_docs])

# Build FAISS index
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)


def re_rank_documents(query, top_k_docs):
    pairs = [[query, doc] for doc in top_k_docs]
    scores = cross_encoder.predict(pairs)
    ranked_docs = [doc for _, doc in sorted(zip(scores, top_k_docs), reverse=True)]
    return ranked_docs

def retrieve_top_k_documents(query_embedding, k=3):
    query_embedding = np.ascontiguousarray(query_embedding, dtype='float32')
    D, I = index.search(query_embedding, k)
    return I  # Return indices of top-k documents

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load T5 model for answer generation
t5_tokenizer = T5Tokenizer.from_pretrained('t5-large')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-large')

def generate_answer(query, top_k_docs):
    # Improve context by concatenating more of the top-ranked documents
    context = " ".join([doc[:512] for doc in top_k_docs])  # Consider more of each document's text

    # Feed the query and improved context to the model
    input_text = f"question: {query} context: {context}"

    inputs = t5_tokenizer(input_text, return_tensors='pt', truncation=True)
    output = t5_model.generate(**inputs)
    answer = t5_tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

import spacy

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

def extract_evidence(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def rag_pipeline(query):
    # Step 1: Encode the query
    query_embedding = encode_query(query)

    # Step 2: Retrieve top-k documents
    top_k_indices = retrieve_top_k_documents(query_embedding, k=2)
    top_k_indices = top_k_indices.flatten()  # Flatten the 2D array to 1D

    # Retrieve the actual documents based on these indices
    top_k_docs = [corpus_docs[i] for i in top_k_indices]

    # Step 3: Re-rank the documents using Cross-Encoder
    top_k_docs = re_rank_documents(query, top_k_docs)

    # Step 4: Generate the answer with better context
    answer = generate_answer(query, top_k_docs)

    # Step 5: Extract evidence
    evidence = []
    for i, doc in enumerate(top_k_docs):
        entities = extract_evidence(doc)
        evidence.append({
            "title": corpus[top_k_indices[i]]['title'],
            "author": corpus[top_k_indices[i]]['author'],
            "url": corpus[top_k_indices[i]]['url'],
            "source": corpus[top_k_indices[i]]['source'],
            "category": corpus[top_k_indices[i]]['category'],
            "published_at": corpus[top_k_indices[i]]['published_at']
        })

    # Format the final output
    result = {
        "query": query,
        "answer": answer,
        "question_type": "inference_query",
        "evidence_list": evidence
    }

    return result



# Test the pipeline
query = "Do the TechCrunch article on software companies and the Hacker News article on The Epoch Times both report an increase in revenue related to payment and subscription models, respectively?"
output = rag_pipeline(query)
print(json.dumps(output, indent=4))

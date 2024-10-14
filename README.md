# RAG Pipeline for Document Retrieval and Answer Generation

---

## Overview

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline designed to enhance question-answering capabilities by leveraging a combination of dense retrieval and generative models. The system allows users to input a query, retrieve relevant documents from a corpus, and generate an answer based on the content of these documents. This approach is particularly effective for complex queries requiring context from multiple sources.

### Problem Statement
In traditional question-answering systems, responses are often generated based on limited context or are entirely reliant on predefined databases. This project aims to overcome these limitations by integrating dense retrieval techniques with powerful generative models, enabling the retrieval of relevant documents and the generation of comprehensive answers.

### Key Features
- **Document Retrieval**: Retrieves relevant documents using embeddings and similarity search.
- **Answer Generation**: Generates coherent answers based on the context provided by retrieved documents.
- **Evidence Extraction**: Identifies and extracts supporting evidence from the retrieved documents.

---

## Folder Structure
```

├── corpus.json          # The dataset containing the documents for retrieval
├── main.py              # Main script implementing the RAG pipeline
├── requirements.txt     # Required libraries for the project
└── README.md            # This README file
           
```
---
## How to Run the Repository

To run the RAG pipeline for document retrieval and answer generation, follow the steps outlined below.

### Step 1: Clone the Repository

Start by cloning the repository to your local machine. Open your terminal or command prompt and run the following command:

```
git clone https://github.com/your-username/rag-pipeline.git
cd rag-pipeline
```
### Step 2: Install Dependencies

```
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset
Place the file in the root directory of the project.

### Step 4: Run the Application
```
python main.py
```
---

## Tech Stack

- *[Hugging Face Transformers]*:
  - Used for query and document encoding with MiniLM and Cross-Encoder re-ranking (models: sentence-transformers/all-MiniLM-L6-v2 and cross-encoder/ms-marco-MiniLM-L-6-v2).

- *[FAISS (Facebook AI Similarity Search)]*:
  - Leveraged for efficient dense retrieval and similarity search between query and document embeddings.

- *[Sentence-Transformers](https://www.sbert.net/)*:
  - Used for encoding both the query and documents into dense vectors using the all-MiniLM-L6-v2 model.

- *[T5 (Text-to-Text Transfer Transformer)](https://huggingface.co/t5-large)*:
  - Employed for generating answers from the retrieved and re-ranked documents (model: t5-large).

- *[spaCy](https://spacy.io/)*:
  - For Named Entity Recognition (NER) to extract relevant entities as evidence from the text.

- *Python Libraries*:
  - *[PyTorch]*: Used for deep learning computations and embeddings.
  - *[NumPy]*: Utilized for matrix operations and handling embeddings.
  - *[JSON]*: For reading and writing corpus and result data.

- *Deployment Tools*:
  - Streamlit for demonstration of our solution through a web application.

---
## Our Solution
## Overview

The RAG Pipeline Solution is designed to enhance the process of information retrieval and answer generation in response to user queries. By leveraging state-of-the-art natural language processing (NLP) techniques, the pipeline efficiently retrieves relevant documents and generates accurate answers. This approach addresses the challenge of information overload by providing users with quick and contextually rich responses.
This solution represents a significant advancement in how we approach natural language understanding and information retrieval, offering a more efficient and effective way for users to access and comprehend information.

## How It Works

The solution comprises several key components that work together to retrieve and process information:

1. *Document Encoding*: 
   - The pipeline uses a pre-trained MiniLM model to transform the documents in the corpus into dense vector representations. This encoding allows for efficient comparison and retrieval based on user queries.

2. *FAISS Indexing*:
   - FAISS (Facebook AI Similarity Search) is employed to index the document embeddings, enabling rapid retrieval of the most relevant documents in response to a query.

3. *Query Encoding*:
   - User queries are encoded into dense vectors using the same MiniLM model. This ensures consistency in the representation of both queries and documents.

4. *Top-K Document Retrieval*:
   - The pipeline retrieves the top-k relevant documents by searching the FAISS index using the encoded query. This step ensures that only the most pertinent documents are considered for further processing.

5. *Document Re-ranking*:
   - A Cross-Encoder model is utilized to re-rank the retrieved documents based on their relevance to the query. This step improves the quality of the results by ensuring that the most contextually relevant documents are prioritized.

6. *Answer Generation*:
   - The T5 model generates an answer by combining the user query with the context of the retrieved and re-ranked documents. This context-aware generation enhances the accuracy of the response.

7. *Evidence Extraction*:
   - Named Entity Recognition (NER) is performed on the top documents to extract relevant entities that support the generated answer. This helps provide additional context and credibility to the response.

## Problem Addressed

In today's digital age, users often face the challenge of navigating vast amounts of information. Traditional search methods can yield numerous results, making it difficult to find precise answers. The RAG pipeline addresses this issue by combining retrieval and generation techniques, enabling users to receive direct, contextually relevant answers quickly. By focusing on both document relevance and contextual understanding, this solution enhances the overall user experience in information retrieval tasks.

### Functions

1. **encode_query(query)**:
   - Encodes the user's query into a dense vector representation using the MiniLM model for efficient retrieval.

2. **embed_document(text)**:
   - Converts document text into vector embeddings using the same model to enable similarity search with FAISS.

3. **retrieve_top_k_documents(query_embedding, k)**:
   - Searches the FAISS index to retrieve the top-k documents most similar to the encoded query.

4. **re_rank_documents(query, top_k_docs)**:
   - Re-ranks the retrieved documents using a Cross-Encoder model to prioritize those most relevant to the query.

5. **generate_answer(query, top_k_docs)**:
   - Generates an answer by using the T5 model, leveraging the query and the context from the top-ranked documents.

6. **extract_evidence(text)**:
   - Uses spaCy to perform Named Entity Recognition (NER) on the text and extract key entities as evidence.

7. **rag_pipeline(query)**:
   - Orchestrates the entire process: encoding the query, retrieving and re-ranking documents, generating an answer, and extracting evidence.
---

## Citations

### Research Papers

1. *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*  
   Authors: Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou  
   Paper: [https://arxiv.org/abs/2002.10957](https://arxiv.org/abs/2002.10957)  
   MiniLM is utilized in this pipeline for both query and document embeddings.

2. *Dense Passage Retrieval (DPR)*  
   Authors: Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih  
   Paper: [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)  
   DPR informs the dense retrieval methodology applied in this solution via FAISS indexing.

3. *T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*  
   Authors: Colin Raffel et al.  
   Paper: [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)  
   The T5 model is used for the generative part of the pipeline to answer user queries.

### Open Source Libraries

1. *Transformers*  
   URL: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)  
   The Hugging Face transformers library powers all the pre-trained models used in this solution, including MiniLM, T5, and the Cross-Encoder.

2. *FAISS (Facebook AI Similarity Search)*  
   URL: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)  
   FAISS enables efficient similarity search for document retrieval in the RAG pipeline.

3. *Sentence Transformers*  
   URL: [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)  
   Sentence Transformers provides the pre-trained all-MiniLM-L6-v2 model for query and document embeddings.

4. *spaCy*  
   URL: [https://github.com/explosion/spaCy](https://github.com/explosion/spaCy)  
   spaCy is utilized for Named Entity Recognition (NER) to extract evidence from the retrieved documents.

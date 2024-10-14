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
└──  requirements.txt     # Required libraries for the project
           
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

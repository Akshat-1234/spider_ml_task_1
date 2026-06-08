# SPML Task 1

This repository contains solutions for both parts of SPML Task 1:

1. **Base Task** – Fashion-MNIST classification using a custom neural network implemented in PyTorch.
2. **Applied ML Domain Task** – Research Paper RAG (Retrieval-Augmented Generation) chatbot built using LangChain, FAISS, Sentence Transformers, Gemini, and Streamlit.

---

## Repository Structure

```text
spider_ml_task_1/
│
├── base_task/
│   ├── SPML_BASE_TASK.ipynb
│   ├── model_weights.pkl
│   ├── submission.csv
│   ├── requirements.txt
│   └── README.md
│
├── applied_ml_domain/
│   ├── app.py
│   ├── rag.py
│   ├── build_vectorstore.py
│   ├── papers/
│   ├── vectorstore/
│   ├── Screenshots/
│   ├── requirements.txt
│   └── README.md
├── .gitigonre
├── requirements.txt
└── README.md
```

---

## Base Task

Implemented the neural network architecture provided in the assignment specification and trained it on the Fashion-MNIST dataset.

**Highlights**

* PyTorch implementation
* Custom dual-branch architecture with skip connection
* Training and validation monitoring
* Performance visualization
* Model weight saving
* Submission file generation

Detailed documentation is available in:

```text
base_task/Readme.md
```

---

## Applied ML Domain Task

Built a Research Paper RAG chatbot capable of answering questions using a collection of research papers.

**Highlights**

* PDF ingestion and text extraction
* Chunking and embedding generation
* FAISS vector database
* Retrieval-Augmented Generation pipeline
* Gemini-powered answer generation
* Source attribution
* Streamlit user interface

Detailed documentation is available in:

```text
applied_ml_domain/README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Applied ML Application

```bash
streamlit run applied_ml_domain/app.py
```

---

## Author

Akshat Goyal

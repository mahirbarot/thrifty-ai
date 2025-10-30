
# Thrifty AI Coding Assessment

This repository contains implementations for two AI/ML tasks: a Retrieval-Augmented QA System and a manual backpropagation implementation for binary classification.

## Task 1: LLM + Retrieval QA System

### Setup
```bash
pip install -r requirements.txt
# i have provided my gemini key.
```

### Design Choices
- **Embeddings**: Used `sentence-transformers` (all-MiniLM-L6-v2) for efficient semantic search
- **Retrieval**: FAISS with cosine similarity (IndexFlatIP with L2 normalization) for fast vector search
- **LLM**: Google Gemini 2.0 Flash for answer generation with temperature=0.3 for consistency
- **Knowledge Base**: 5+ text files in `/data` folder covering various technical topics. (You will have to upload it manually to the collab , i have provided the data folder )

### Confidence Score Calculation
Confidence = (avg_similarity × 0.7) + (score_gap × 0.3), where score_gap is the difference between top-2 retrieved docs. Weighted average accounts for both retrieval quality and ranking certainty.

### Future Improvements
- Implement chunk-based retrieval for larger documents
- Add query expansion and reranking mechanisms
- Integrate evaluation metrics (RAGAS, context relevance)
- Implement caching and async processing for better performance

## Task 2: Manual Backpropagation (2-Layer Binary Classifier)

### Setup
```bash
# NumPy only - no additional dependencies
python task_2.py  # or run task_2.ipynb
```

### Architecture
- Input: 3D vector
- Hidden: 4 units with ReLU activation
- Output: 1 unit with sigmoid activation
- Loss: Binary Cross-Entropy

### Key Implementation Details
- Forward pass computes z1, a1, z2, a2, and BCE loss
- Backward pass derives analytical gradients using chain rule:
  - Output: dL/dz2 = a2 - y (simplified for sigmoid + BCE)
  - Hidden: dz1 = (W2^T · dz2) ⊙ ReLU'(z1)
- Numerical gradient check validates analytical gradients (max diff < 1e-7)
- SGD update with lr=0.1, fixed seed (42) for reproducibility

### Output
The script prints initial loss, analytical gradients (dW1, dW2), gradient check differences, updated weights, and final loss (which decreases after SGD step).

## Repository Structure
```
├── data/                    # Text documents for Task 1
├── task_1_rag.ipynb        # RAG QA System implementation
├── task_2.ipynb            # Manual backprop implementation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Author
Mahir Barot

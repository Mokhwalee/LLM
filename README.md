# RAG + PEFT LLM Application

This repository focuses on the applications of **Retrieval-Augmented Generation (RAG)** and **Parameter-Efficient Fine-Tuning (PEFT)**.  
It highlights my work in applying these techniques to build practical, domain-adapted large language models, with a focus on **customer support automation**.

---

## Objectives

- Explore **retrieval-based NLP pipelines** (document indexing, embedding models, similarity search).  
- Apply **efficient fine-tuning techniques** (LoRA, quantization) to large models.  
- Integrate **RAG + PEFT** into an end-to-end chatbot system.  
- Evaluate performance with **automated metrics** and **qualitative analysis**.  
- Build a **deployable prototype** with a simple user interface.  

---

## Repository Structure

```
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_vector_search.ipynb
│   ├── 04_peft_lora.ipynb
│   ├── 05_rag_pipeline.ipynb
│   ├── 06_evaluation.ipynb
│   ├── 07_web_app.ipynb
│   └── 08_final_project.ipynb
│
├── data/              # Example datasets (customer support logs, Kaggle samples, etc.)
├── utils/             # Helper functions (retrievers, training, evaluation)
├── reports/           # Notes, reflections, and findings
├── requirements.txt
└── README.md
```

---

## Study Sections

### 1. Data Exploration & Preprocessing  
- Collecting and cleaning open-source datasets.  
- Building preprocessing pipelines for text normalization, tokenization, and filtering.  

### 2. Vector Search Systems  
- Implementing embedding models (e.g., `sentence-transformers`).  
- Building vector indexes with **FAISS** and **Chroma**.  
- Running similarity search experiments.  

### 3. Parameter-Efficient Fine-Tuning (PEFT)  
- Exploring **LoRA adapters** for large LLMs.  
- Quantization techniques for resource efficiency.  
- Training on domain-specific QA datasets.  

### 4. Retrieval-Augmented Generation (RAG)  
- Implementing retriever → generator workflows.  
- Context-aware prompt design.  
- Testing retrieval-augmented outputs vs. base LLM.  

### 5. Evaluation  
- Retrieval metrics: **recall@k, MRR**.  
- Generation metrics: **Exact Match, F1, Rouge-L**.  
- Experimenting with **LLM-as-a-judge evaluation**.  

### 6. Web Interface & Deployment  
- Building a lightweight prototype with **Streamlit** or **FastAPI**.  
- Integrating RAG + PEFT into a chatbot interface.  
- Deploying for interactive demos.  

---

## Tech Stack

- **Core Libraries**: `PyTorch`, `transformers`, `peft`, `datasets`  
- **Retrieval**: `faiss-cpu`, `chromadb`, `sentence-transformers`  
- **App Layer**: `streamlit`, `fastapi`  
- **Environment**: Google Colab / Jupyter Notebook  

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Quick Demo

```python
from utils.indexing import build_index
from utils.retrievers import Retriever
from utils.peft_utils import load_lora_model

# Build retriever
index = build_index("data/knowledge_base")
retriever = Retriever(index)

# Load LoRA fine-tuned model
model, tokenizer = load_lora_model(
    base_model="meta-llama/Llama-2-7b-hf",
    adapter_path="checkpoints/lora"
)

query = "How can PEFT improve customer support chatbots?"
context = retriever(query, top_k=3)
answer = model.generate_with_context(query, context)

print(answer)
```

---

## Outcomes

- A working **retrieval-augmented chatbot**.  
- Hands-on experience with **LoRA fine-tuning**.  
- Deployed prototype demonstrating practical **LLM applications**.  
- Portfolio project showing applied skills in **modern NLP systems**.  

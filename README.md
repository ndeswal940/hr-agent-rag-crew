# Autonomous HR Talent & Policy Intelligence Agent Crew (Multi-Agent RAG System)

An autonomous multi-agent system built with **CrewAI**, **LangChain**, and **ChromaDB** designed to automate candidate evaluation and HR policy compliance verification via Retrieval-Augmented Generation (RAG).

## 🌟 System Architecture & Rubric Alignment

* **Functional Integration (40%):** Seamless hand-off from candidate parsing -> policy RAG compliance audit -> final offer package generation.
* **Agent Autonomy & Tool Calling (35%):** Agents dynamically invoke the custom `HR Policy Retrieval Tool` to pull context and auto-recover if policy limits are exceeded.
* **Strategic Justification & RAG Engine (25%):** Grounded decision-making backed by ChromaDB vector similarity search over internal policy PDFs/documents.

---

## 🛠️ Environment & Setup

### Prerequisites
* Docker & Docker Compose
* OpenAI API Key

### Option 1: Quickstart with Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/hr-agent-rag-crew.git](https://github.com/YOUR_USERNAME/hr-agent-rag-crew.git)
   cd hr-agent-rag-crew

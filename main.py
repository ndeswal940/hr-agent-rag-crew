import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool
from crewai import Agent, Task, Crew, Process

# Set your API keys (or pass via environment variables)
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# -------------------------------------------------------------------
# 1. RAG INGESTION & VECTOR STORE SETUP (25% Weightage Component)
# -------------------------------------------------------------------
def initialize_rag_database():
    """Builds persistent local ChromaDB instance with HR documents."""
    sample_policy = """
    COMPANY HR & COMPENSATION POLICY 2026:
    1. Remote Work: Senior Engineers (Level 4+) are eligible for 100% remote work. Junior/Mid (Level 1-3) require hybrid (2 days in-office).
    2. Signing Bonus: Max signing bonus for Level 4 is $15,000. Level 5+ can go up to $30,000.
    3. Notice Period: Standard notice period is 30 days. Exceptions require VP approval.
    """
    
    with open("hr_policy.txt", "w") as f:
        f.write(sample_policy)

    loader = TextLoader("hr_policy.txt")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

# Build RAG Database
vectorstore = initialize_rag_database()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# -------------------------------------------------------------------
# 2. CUSTOM RAG TOOL FOR AGENT AUTONOMY
# -------------------------------------------------------------------
@tool("HR Policy Retrieval Tool")
def query_hr_policy(query: str) -> str:
    """Queries the internal HR policy vector database to retrieve company guidelines and compliance constraints."""
    results = retriever.invoke(query)
    if not results:
        return "ERROR_RETRIEVAL_FAILED: No matching HR policy context found. Fall back to standard conservative guidelines."
    
    context = "\n---\n".join([doc.page_content for doc in results])
    return f"RETRIEVED POLICY CONTEXT:\n{context}"

# -------------------------------------------------------------------
# 3. CREWAI AGENT ORCHESTRATION
# -------------------------------------------------------------------
policy_analyst = Agent(
    role="HR Policy & Compliance Specialist",
    goal="Ensure all hiring actions and compensation proposals strictly comply with company policies.",
    backstory="You are a meticulous HR auditor who cross-checks all compensation and remote work requests against corporate policy using vector store tools.",
    tools=[query_hr_policy],
    verbose=True,
    memory=True,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

talent_evaluator = Agent(
    role="Senior Technical Recruiter",
    goal="Evaluate candidate qualifications and determine candidate seniority level.",
    backstory="You are an expert tech recruiter who assesses candidate experience to assign seniority levels and fit.",
    verbose=True,
    memory=True,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
)

# Tasks Definition
task1 = Task(
    description="Evaluate candidate profile: 'John Doe, 8 years Senior Backend Engineer experience requesting 100% remote work and $20,000 signing bonus.' Determine seniority level.",
    expected_output="Detailed evaluation with assigned level (e.g., Level 4 or Level 5).",
    agent=talent_evaluator
)

task2 = Task(
    description="Use the HR Policy Retrieval Tool to check if John Doe's requests (100% remote work, $20,000 signing bonus) comply with policy for his assigned level.",
    expected_output="Compliance report detailing policy alignment, allowed max bonus, and remote work eligibility.",
    agent=policy_analyst
)

# Crew Assembly
hr_crew = Crew(
    agents=[talent_evaluator, policy_analyst],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    result = hr_crew.kickoff()
    print("\n================ FINAL AGENT CREW OUTPUT ================\n")
    print(result)

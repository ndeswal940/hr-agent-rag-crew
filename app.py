import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.tools import tool
from crewai import Agent, Task, Crew, Process

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION & SECRETS INITIALIZATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="HR Talent & Policy Intelligence Agent Crew",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Autonomous HR Talent & Policy Intelligence Crew")
st.caption("Multi-Agent Architecture powered by Groq API (Llama 3.3 70B) & ChromaDB RAG Engine")

# Safe API key initialization for Render and local development
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    try:
        if "GROQ_API_KEY" in st.secrets:
            groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY is missing! Please configure environment variables in Render.")
    st.stop()
# -------------------------------------------------------------------
# 2. RAG VECTOR STORE INITIALIZATION (Cached for Performance)
# -------------------------------------------------------------------
@st.cache_resource
def load_rag_retriever():
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

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = load_rag_retriever()

# -------------------------------------------------------------------
# 3. CUSTOM RAG TOOL
# -------------------------------------------------------------------
@tool("HR Policy Retrieval Tool")
def query_hr_policy(query: str) -> str:
    """Queries the internal HR policy vector database to retrieve company guidelines and compliance constraints."""
    results = retriever.invoke(query)
    if not results:
        return "ERROR_RETRIEVAL_FAILED: No matching HR policy context found."
    
    context = "\n---\n".join([doc.page_content for doc in results])
    return f"RETRIEVED POLICY CONTEXT:\n{context}"

# -------------------------------------------------------------------
# 4. STREAMLIT USER INTERFACE & CREW EXECUTION
# -------------------------------------------------------------------
st.sidebar.header("📋 Candidate Input Profile")
candidate_name = st.sidebar.text_input("Candidate Name", "John Doe")
experience_years = st.sidebar.number_input("Years of Experience", min_value=1, max_value=30, value=8)
requested_role = st.sidebar.text_input("Role Requested", "Senior Backend Engineer")
requested_remote = st.sidebar.selectbox("Remote Request", ["100% Remote Work", "Hybrid (2 days office)", "On-site"])
requested_bonus = st.sidebar.number_input("Requested Signing Bonus ($)", min_value=0, max_value=50000, value=20000, step=1000)

if st.sidebar.button("🚀 Run Agent Evaluation Crew"):
    with st.spinner("Multi-Agent Crew is evaluating candidate and auditing HR policy..."):
        
        # Initialize Groq LLM
        groq_llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0.1
        )

        # Initialize Agents
        talent_evaluator = Agent(
            role="Senior Technical Recruiter",
            goal="Evaluate candidate qualifications and assign seniority level.",
            backstory="You assess candidate experience to assign seniority levels (Level 1 to Level 5).",
            verbose=True,
            memory=True,
            llm=groq_llm
        )

        policy_analyst = Agent(
            role="HR Policy & Compliance Specialist",
            goal="Ensure hiring proposals strictly comply with company policy via RAG retrieval.",
            backstory="You audit requests against corporate policy using the HR Policy Retrieval Tool.",
            tools=[query_hr_policy],
            verbose=True,
            memory=True,
            llm=groq_llm
        )

        offer_strategist = Agent(
            role="Compensation & Offer Strategist",
            goal="Synthesize recruitment evaluations and compliance checks into a final offer package.",
            backstory="You produce final, policy-compliant candidate offer decisions.",
            verbose=True,
            memory=True,
            llm=groq_llm
        )

        # Define Tasks
        candidate_prompt = f"{candidate_name}, {experience_years} years experience as {requested_role}, requesting {requested_remote} and ${requested_bonus:,} signing bonus."

        task1 = Task(
            description=f"Evaluate candidate profile: '{candidate_prompt}'. Assign seniority level.",
            expected_output="Seniority level determination (e.g., Level 4).",
            agent=talent_evaluator
        )

        task2 = Task(
            description=f"Use the HR Policy Retrieval Tool to audit if {requested_remote} and ${requested_bonus:,} bonus comply with policy for assigned level.",
            expected_output="Compliance report detailing remote eligibility and max bonus allowed.",
            agent=policy_analyst
        )

        task3 = Task(
            description="Synthesize evaluation and policy compliance findings to produce final offer recommendation.",
            expected_output="Final structured offer decision specifying approved remote status and final approved bonus amount.",
            agent=offer_strategist
        )

        # Execute Crew
        hr_crew = Crew(
            agents=[talent_evaluator, policy_analyst, offer_strategist],
            tasks=[task1, task2, task3],
            process=Process.sequential,
            verbose=True
        )

        final_output = hr_crew.kickoff()

        st.success("✅ Multi-Agent Evaluation Complete!")
        
        st.subheader("📄 Final Agent Crew Decision Package")
        st.markdown(final_output)

        with st.expander("🔍 View Policy Database Context (RAG Ingestion)"):
            st.code("""
COMPANY HR & COMPENSATION POLICY 2026:
1. Remote Work: Senior Engineers (Level 4+) are eligible for 100% remote work. Junior/Mid (Level 1-3) require hybrid (2 days in-office).
2. Signing Bonus: Max signing bonus for Level 4 is $15,000. Level 5+ can go up to $30,000.
            """, language="text")

import os
import streamlit as st

# ChromaDB / SQLite patch for Linux environments (Render)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Page Setup
st.set_page_config(page_title="HR Talent & Policy Intelligence Agent Crew", page_icon="🤖", layout="wide")
st.title("🤖 Autonomous HR Talent & Policy Intelligence Crew")
st.caption("Custom Multi-Agent Architecture powered by Groq API & RAG Engine")

# -------------------------------------------------------------------
# 1. SIDEBAR CONFIGURATION & API KEY INPUT
# -------------------------------------------------------------------
st.sidebar.header("🔑 API Configuration")

show_key = st.sidebar.checkbox("Show API Key", value=False)
user_api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="default" if show_key else "password",
    help="Paste your Groq API key here (starts with gsk_)"
)

# Priority: UI Sidebar Input -> Render Environment Variable -> Streamlit Secrets
groq_api_key = user_api_key or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    try:
        if "GROQ_API_KEY" in st.secrets:
            groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

# -------------------------------------------------------------------
# 2. FAST LIGHTWEIGHT RAG RETRIEVER INITIALIZATION
# -------------------------------------------------------------------
sample_policy = """
COMPANY HR & COMPENSATION POLICY 2026:
1. Remote Work: Senior Engineers (Level 4+) are eligible for 100% remote work. Junior/Mid (Level 1-3) require hybrid (2 days in-office).
2. Signing Bonus: Max signing bonus for Level 4 is $15,000. Level 5+ can go up to $30,000.
3. Notice Period: Standard notice period is 30 days. Exceptions require VP approval.
"""

if not os.path.exists("hr_policy.txt"):
    with open("hr_policy.txt", "w") as f:
        f.write(sample_policy)

loader = TextLoader("hr_policy.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

retriever = BM25Retriever.from_documents(chunks)
retriever.k = 2

def query_hr_policy(query: str) -> str:
    """Queries internal HR policy vector database."""
    results = retriever.invoke(query)
    if not results:
        return "ERROR_RETRIEVAL_FAILED: No matching HR policy context found."
    return "\n---\n".join([doc.page_content for doc in results])

# -------------------------------------------------------------------
# 3. SIDEBAR USER INPUTS
# -------------------------------------------------------------------
st.sidebar.header("📋 Candidate Input Profile")
name = st.sidebar.text_input("Candidate Name", "John Doe")
experience = st.sidebar.number_input("Years of Experience", min_value=1, max_value=30, value=8)
role = st.sidebar.text_input("Role Requested", "Senior Backend Engineer")
remote_req = st.sidebar.selectbox("Remote Request", ["100% Remote Work", "Hybrid (2 days office)", "On-site"])
bonus_req = st.sidebar.number_input("Requested Signing Bonus ($)", min_value=0, max_value=50000, value=20000, step=1000)

run_button = st.sidebar.button("🚀 Run Agent Evaluation")

if not groq_api_key:
    st.warning("👈 Pehle sidebar me apni **Groq API Key** enter karein taaki execution start ho sake.")
else:
    st.success("✅ Groq API Key Configured Successfully!")

# -------------------------------------------------------------------
# 4. SEQUENTIAL AGENT PIPELINE EXECUTION
# -------------------------------------------------------------------
if run_button:
    if not groq_api_key:
        st.error("⚠️ Please enter a valid Groq API Key in the sidebar before running.")
    else:
        with st.spinner("Running Multi-Agent Evaluation Sequence..."):
            
            # Initialize LLM directly via ChatGroq
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="openai/gpt-oss-120b",
                temperature=0.1
            )

            # --- AGENT 1: Senior Technical Recruiter ---
            recruiter_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a Senior Technical Recruiter. Evaluate candidate experience and assign a seniority level (Level 1 to Level 5). Provide brief reasoning."),
                ("human", "Candidate Details: Name: {name}, Experience: {exp} years, Requested Role: {role}")
            ])
            recruiter_chain = recruiter_prompt | llm
            eval_result = recruiter_chain.invoke({
                "name": name,
                "exp": experience,
                "role": role
            }).content

            st.subheader("Step 1: 🎯 Recruiter Evaluation")
            st.info(eval_result)

            # --- AGENT 2: HR Policy Specialist (with RAG Retrieval) ---
            policy_context = query_hr_policy(f"{role} remote work bonus level policy")
            
            policy_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an HR Policy Specialist. Audit requested terms against corporate guidelines retrieved via RAG.\n\nRetrieved Policy Guidelines:\n{context}"),
                ("human", "Recruiter Evaluation Output:\n{eval_output}\n\nCandidate Requests:\nRemote Request: {remote}\nSigning Bonus Request: ${bonus}")
            ])
            policy_chain = policy_prompt | llm
            policy_result = policy_chain.invoke({
                "context": policy_context,
                "eval_output": eval_result,
                "remote": remote_req,
                "bonus": bonus_req
            }).content

            st.subheader("Step 2: ⚖️️ HR Policy Audit (RAG Engine)")
            st.warning(policy_result)

            # --- AGENT 3: Offer Strategist ---
            strategist_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an Offer Strategist. Synthesize candidate evaluations and HR policy audit reports into a final executive offer decision package."),
                ("human", "Recruiter Evaluation:\n{eval_output}\n\nPolicy Compliance Audit:\n{audit_output}\n\nCreate a final formatted hiring offer recommendation.")
            ])
            strategist_chain = strategist_prompt | llm
            final_offer = strategist_chain.invoke({
                "eval_output": eval_result,
                "audit_output": policy_result
            }).content

            st.success("✅ Multi-Agent Pipeline Execution Complete!")
            st.subheader("📜 Final Executive Offer Decision Package")
            st.markdown(final_offer)

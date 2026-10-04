import os
import re
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
st.set_page_config(page_title="Indian HR Compensation & Policy AI Crew", page_icon="🇮🇳", layout="wide")
st.title("🇮🇳 Autonomous Indian HR Compensation & Policy Intelligence Crew")
st.caption("Multi-Agent Architecture powered by Groq API & RAG Engine (Tailored for Indian Salary Standards)")

# Helper function to clean text rendering and ensure correct formatting
def clean_output(text: str) -> str:
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = text.replace('∗∗', '**')
    text = re.sub(r'(\d[\d,]*)([A-Za-z\(])', r'\1 \2', text)
    return text

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

groq_api_key = user_api_key or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    try:
        if "GROQ_API_KEY" in st.secrets:
            groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

# -------------------------------------------------------------------
# 2. INDIAN HR POLICY & RAG RETRIEVER INITIALIZATION
# -------------------------------------------------------------------
sample_indian_policy = """
INDIAN HR & COMPENSATION POLICY GUIDELINES (2026):
1. Level Hierarchy & Compensation (CTC in INR):
   - Level 1-2 (Software Engineer / Mid): Base CTC Band ₹6 LPA to ₹14 LPA. Max joining/joining bonus capped at ₹1,00,000.
   - Level 3-4 (Senior Engineer / Tech Lead): Base CTC Band ₹15 LPA to ₹28 LPA. Max joining bonus capped at ₹2,50,000.
   - Level 5+ (Engineering Manager / Staff Engineer): Base CTC Band ₹30 LPA+. Max joining bonus capped at ₹5,00,000.

2. Work Model Policy:
   - Level 1-3: Hybrid mandatory (Min 3 days office attendance in Indian Tech Hubs: Bangalore, Gurgaon, Pune, Hyderabad).
   - Level 4+: Eligible for 100% Work From Home (WFH) / Remote option upon VP/HR Approval.

3. Standard Indian Salary Structure Components:
   - Basic Salary: 40% to 50% of Fixed CTC.
   - House Rent Allowance (HRA): 50% of Basic for Metro cities (Delhi-NCR, Bangalore, Mumbai), 40% for Non-Metro.
   - Special Allowance / Flexible Benefit Basket (FBA): Remaining Fixed Balance.
   - Employer PF Contribution: 12% of Basic (Deducted towards Statutory EPF).

4. Notice Period & Retention:
   - Standard Notice Period in India: 60 days (or 90 days for Critical Tech roles). Buyout/Reduction requires VP HR sign-off.
"""

if not os.path.exists("indian_hr_policy.txt"):
    with open("indian_hr_policy.txt", "w") as f:
        f.write(sample_indian_policy)

loader = TextLoader("indian_hr_policy.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

retriever = BM25Retriever.from_documents(chunks)
retriever.k = 2

def query_hr_policy(query: str) -> str:
    """Queries internal Indian HR policy vector database."""
    results = retriever.invoke(query)
    if not results:
        return "ERROR_RETRIEVAL_FAILED: No matching HR policy context found."
    return "\n---\n".join([doc.page_content for doc in results])

# -------------------------------------------------------------------
# 3. SIDEBAR USER INPUTS (INDIAN CURRENCY & SALARY FORMAT)
# -------------------------------------------------------------------
st.sidebar.header("📋 Candidate Input Profile")
name = st.sidebar.text_input("Candidate Name", "Aarav Sharma")
experience = st.sidebar.number_input("Years of Experience", min_value=1, max_value=30, value=7)
role = st.sidebar.text_input("Role Requested", "Senior Backend Engineer")
expected_ctc = st.sidebar.number_input("Expected Fixed CTC (in ₹ LPA)", min_value=3.0, max_value=100.0, value=22.0, step=0.5)
remote_req = st.sidebar.selectbox("Work Location Request", ["100% Remote / WFH", "Hybrid (3 days office)", "On-site (Bangalore/Gurgaon)"])
bonus_req = st.sidebar.number_input("Requested Joining Bonus (in ₹ INR)", min_value=0, max_value=1000000, value=300000, step=25000)

run_button = st.sidebar.button("🚀 Evaluate Candidate Offer Package")

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
        with st.spinner("Running Indian Multi-Agent Evaluation Sequence..."):
            
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="openai/gpt-oss-120b",
                temperature=0.1
            )

            # --- AGENT 1: Senior Technical Recruiter (India Tech Hub) ---
            recruiter_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a Senior Technical Recruiter specializing in the Indian IT Industry. Evaluate candidate experience, role requested, and expected CTC (in ₹ LPA). Assign a seniority level (Level 1 to Level 5) with clear justification in clean Markdown format with proper spacing."),
                ("human", "Candidate Details: Name: {name}, Experience: {exp} years, Requested Role: {role}, Expected CTC: ₹{ctc} LPA")
            ])
            recruiter_chain = recruiter_prompt | llm
            eval_result = clean_output(recruiter_chain.invoke({
                "name": name,
                "exp": experience,
                "role": role,
                "ctc": expected_ctc
            }).content)

            st.subheader("Step 1: 🎯 Indian Talent Recruiter Evaluation")
            st.info(eval_result)

            # --- AGENT 2: HR Policy Specialist (Indian RAG Engine) ---
            policy_context = query_hr_policy(f"{role} remote work joining bonus CTC Indian policy")
            
            policy_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an Indian HR Policy Specialist. Audit requested compensation terms (CTC in ₹ LPA, Joining Bonus in ₹, and Work location) against corporate guidelines retrieved via RAG.\n\nRetrieved Policy Guidelines:\n{context}\n\nSTRICT FORMATTING RULES: Do not use HTML tags like <br>. Use clean Markdown bullets and tables. Ensure correct spacing for Indian Currency numbers (₹)."),
                ("human", "Recruiter Evaluation Output:\n{eval_output}\n\nCandidate Requests:\nExpected Fixed CTC: ₹{ctc} LPA\nWork Location Request: {remote}\nJoining Bonus Request: ₹{bonus:,}")
            ])
            policy_chain = policy_prompt | llm
            policy_result = clean_output(policy_chain.invoke({
                "context": policy_context,
                "eval_output": eval_result,
                "ctc": expected_ctc,
                "remote": remote_req,
                "bonus": bonus_req
            }).content)

            st.subheader("Step 2: ⚖️ HR Policy & Compliance Audit (Indian RAG Engine)")
            st.warning(policy_result)

            # --- AGENT 3: Compensation & Offer Strategist (India Structure) ---
            strategist_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an Executive Compensation Strategist in India. Synthesize candidate evaluations and HR policy audit reports into a final Indian Hiring Offer Decision Package.

Provide a complete Indian CTC Salary Component Breakdown table:
- Base Fixed CTC (₹ LPA)
- Basic Salary (50% of CTC)
- HRA (50% of Basic for Metro)
- Special Allowance / FBA (Balance)
- Employer PF Contribution (12% of Basic)
- Joining Bonus (Approved Amount in ₹)
- Work Location & Notice Period (60 days standard)

STRICT FORMATTING RULES: Use standard Markdown tables and bullet points. Do NOT use HTML tags like <br>. Ensure proper spacing between numbers, symbols (₹), and words."""),
                ("human", "Recruiter Evaluation:\n{eval_output}\n\nPolicy Compliance Audit:\n{audit_output}\n\nCreate a final formatted hiring offer recommendation suited for Indian market standards.")
            ])
            strategist_chain = strategist_prompt | llm
            final_offer = clean_output(strategist_chain.invoke({
                "eval_output": eval_result,
                "audit_output": policy_result
            }).content)

            st.success("✅ Multi-Agent Pipeline Execution Complete!")
            st.subheader("📜 Final Executive Offer Decision Package (Indian Compensation Standards)")
            st.markdown(final_offer)

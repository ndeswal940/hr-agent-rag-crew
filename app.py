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
from langchain.tools import tool
from crewai import Agent, Task, Crew, Process, LLM

# Page Setup
st.set_page_config(page_title="HR Talent & Policy Intelligence Agent Crew", page_icon="🤖", layout="wide")
st.title("🤖 Autonomous HR Talent & Policy Intelligence Crew")
st.caption("Multi-Agent Architecture powered by Groq API (openai/gpt-oss-120b) & RAG Engine")

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

@tool("HR Policy Retrieval Tool")
def query_hr_policy(query: str) -> str:
    """Queries the internal HR policy vector database to retrieve company guidelines."""
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

run_button = st.sidebar.button("🚀 Run CrewAI Evaluation")

if not groq_api_key:
    st.warning("👈 Pehle sidebar me apni **Groq API Key** enter karein taaki execution start ho sake.")
else:
    st.success("✅ Groq API Key Configured Successfully!")

# -------------------------------------------------------------------
# 4. CREWAI EXECUTION (ONLY RUNS WHEN BUTTON IS CLICKED)
# -------------------------------------------------------------------
if run_button:
    if not groq_api_key:
        st.error("⚠️ Please enter a valid Groq API Key in the sidebar before running.")
    else:
        with st.spinner("Executing CrewAI Agents..."):
            os.environ["GROQ_API_KEY"] = groq_api_key

            # Explicit LLM Instance
            groq_llm = LLM(
                model="groq/openai/gpt-oss-120b",
                api_key=groq_api_key,
                temperature=0.1
            )

            talent_evaluator = Agent(
                role="Senior Technical Recruiter",
                goal="Evaluate candidate qualifications and assign seniority level.",
                backstory="You assess experience to assign seniority levels (Level 1 to 5).",
                verbose=True,
                memory=True,
                llm=groq_llm
            )

            policy_analyst = Agent(
                role="HR Policy Specialist",
                goal="Ensure hiring proposals strictly comply with company policy via RAG.",
                backstory="You audit requests against corporate policy using the HR Policy Retrieval Tool.",
                tools=[query_hr_policy],
                verbose=True,
                memory=True,
                llm=groq_llm
            )

            offer_strategist = Agent(
                role="Offer Strategist",
                goal="Synthesize recruitment evaluations and compliance checks into a final offer package.",
                backstory="You produce final, policy-compliant offer decisions.",
                verbose=True,
                memory=True,
                llm=groq_llm
            )

            candidate_prompt = f"{name}, {experience} years experience as {role}, requesting {remote_req} and ${bonus_req:,} signing bonus."

            task1 = Task(description=f"Evaluate: '{candidate_prompt}'. Assign seniority level.", expected_output="Level assignment.", agent=talent_evaluator)
            task2 = Task(description=f"Use HR Policy Retrieval Tool to audit if {remote_req} and ${bonus_req:,} bonus comply for assigned level.", expected_output="Compliance report.", agent=policy_analyst)
            task3 = Task(description="Synthesize evaluation and policy findings into final offer decision.", expected_output="Final offer package.", agent=offer_strategist)

            hr_crew = Crew(agents=[talent_evaluator, policy_analyst, offer_strategist], tasks=[task1, task2, task3], process=Process.sequential, verbose=True)
            final_output = hr_crew.kickoff()

            st.success("✅ CrewAI Execution Complete!")
            st.subheader("📜 Final Agent Decision Package")
            st.markdown(final_output)

import os
import re
import streamlit as st

# ChromaDB / SQLite patch for Linux environments (Render / Streamlit Cloud)
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
st.set_page_config(page_title="Multi-Industry HR Compensation AI Crew", page_icon="🏢", layout="wide")
st.title("🏢 Multi-Industry HR Compensation & Policy Intelligence Crew")
st.caption("Autonomous Multi-Agent Architecture Supporting 10 Industry Sectors in India")

# Clean formatting helper
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
# 2. DYNAMIC MULTI-INDUSTRY POLICY ENGINE (10 SECTORS)
# -------------------------------------------------------------------
st.sidebar.header("🏭 Industry Sector Selection")
industry = st.sidebar.selectbox(
    "Select Target Industry Sector",
    [
        "Information Technology (IT & Tech)",
        "Automotive & Manufacturing",
        "Banking & Finance (BFSI)",
        "Healthcare & Pharma",
        "E-Commerce & Logistics",
        "FMCG & Consumer Goods",
        "Core Engineering & Construction",
        "Telecommunications",
        "Media & Advertising",
        "Renewable Energy & CleanTech"
    ]
)

INDUSTRY_POLICIES = {
    "Information Technology (IT & Tech)": """
    IT & TECH HR POLICY GUIDELINES:
    - Compensation: CTC Range ₹6 LPA to ₹45 LPA+. Max Joining Bonus ₹2.5L for Sr Engineers.
    - Work Model: Hybrid / 100% WFH eligible for Senior Roles (Level 4+).
    - Allowances: Internet, Home-Office Setup, Night Shift Allowances.
    - Notice Period: Standard 60 Days.
    """,
    "Automotive & Manufacturing": """
    AUTOMOTIVE & MANUFACTURING HR POLICY GUIDELINES:
    - Compensation: CTC Range ₹4.5 LPA to ₹30 LPA. Max Joining Bonus ₹1.0L (Base salary focused).
    - Work Model: 100% On-Site / Plant Mandatory for Operations. Remote restricted to HQ strategy roles.
    - Allowances: Rotational Shift Allowance, Plant Transport, Safety Equipment & Uniform Allowance, Relocation Allowance.
    - Notice Period: Standard 30 to 45 Days.
    """,
    "Banking & Finance (BFSI)": """
    BANKING & FINANCE HR POLICY GUIDELINES:
    - Compensation: Fixed Base + High Performance Variable Bonus (Up to 30% of CTC).
    - Work Model: On-site / Hybrid (Max 2 days office WFH due to compliance and regulatory security).
    - Allowances: Financial Compliance Allowance, Meal Vouchers, Comprehensive Life/Health Insurance.
    - Notice Period: Standard 60 to 90 Days.
    """,
    "Healthcare & Pharma": """
    HEALTHCARE & PHARMA HR POLICY GUIDELINES:
    - Compensation: CTC Range ₹5 LPA to ₹35 LPA. On-call retention bonuses available.
    - Work Model: On-site / Hospital / Lab compulsory for Clinical/R&D roles. WFH allowed for Health-Tech/Data roles.
    - Allowances: Professional Hazard Allowance, On-call Allowance, CME (Continuing Medical Education) Reimbursement.
    - Notice Period: Standard 30 Days.
    """,
    "E-Commerce & Logistics": """
    E-COMMERCE & LOGISTICS HR POLICY GUIDELINES:
    - Compensation: CTC Range ₹4 LPA to ₹28 LPA. Peak season performance incentives available.
    - Work Model: On-site mandatory for Fulfillment/Warehouse managers. Hybrid for Tech/Product roles.
    - Allowances: Night Shift, Fulfillment Center Allowance, Fleet Management Travel Reimbursements.
    - Notice Period: Standard 30 to 45 Days.
    """,
    "FMCG & Consumer Goods": """
    FMCG & CONSUMER GOODS HR POLICY GUIDELINES:
    - Compensation: CTC Range ₹5 LPA to ₹32 LPA. Strong performance-linked quarterly sales incentives.
    - Work Model: Field-based / On-site mandatory for Sales & Operations. Hybrid for Brand/Corporate roles.
    - Allowances: Daily Travel Allowance (TA/DA), Vehicle Maintenance, Regional Mobility Allowance.
    - Notice Period: Standard 30 to 60 Days.
    """,
    "Core Engineering & Construction": """
    CORE ENGINEERING & CONSTRUCTION HR POLICY GUIDELINES:
    - Compensation: CTC Range ₹4.5 LPA to ₹28 LPA. Project completion bonuses applicable.
    - Work Model: 100% Site Location Mandatory for Project Engineers. Corporate HQ allows Hybrid.
    - Allowances: Site Project Hardship Allowance, Free Accommodation / HRA, Safety Equipment Allowance.
    - Notice Period: Standard 30 Days.
    """,
    "Telecommunications": """
    TELECOMMUNICATIONS HR POLICY GUIDELINES:
    - Compensation: CTC Range ₹5 LPA to ₹30 LPA. Network availability performance bonuses.
    - Work Model: On-site/Field for Network Engineers. Hybrid for IT & Corporate Telecom roles.
    - Allowances: Field Duty Mobile/Data Allowance, Night Shift Allowance, Emergency Response Allowance.
    - Notice Period: Standard 60 Days.
    """,
    "Media & Advertising": """
    MEDIA & ADVERTISING HR POLICY GUIDELINES:
    - Compensation: CTC Range ₹4 LPA to ₹25 LPA. Campaign-based incentive bonuses.
    - Work Model: Flexible Hybrid / Studio On-site depending on production schedules.
    - Allowances: Equipment Allowance, Production Meal Vouchers, Overtime Compensation.
    - Notice Period: Standard 30 Days.
    """,
    "Renewable Energy & CleanTech": """
    RENEWABLE ENERGY & CLEANTECH HR POLICY GUIDELINES:
    - Compensation: CTC Range ₹5 LPA to ₹32 LPA. Green Tech innovation retention bonuses.
    - Work Model: On-site Solar/Wind Plant location for Site Engineers. Hybrid for R&D/HQ.
    - Allowances: Remote Site Hardship Allowance, Travel Allowance, Safety Equipment Coverage.
    - Notice Period: Standard 30 to 45 Days.
    """
}

# Load active selected industry policy
selected_policy_text = INDUSTRY_POLICIES[industry]
policy_filename = "active_industry_policy.txt"
with open(policy_filename, "w") as f:
    f.write(selected_policy_text)

loader = TextLoader(policy_filename)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

retriever = BM25Retriever.from_documents(chunks)
retriever.k = 2

def query_hr_policy(query: str) -> str:
    results = retriever.invoke(query)
    if not results:
        return "ERROR_RETRIEVAL_FAILED: No matching HR policy context found."
    return "\n---\n".join([doc.page_content for doc in results])

# -------------------------------------------------------------------
# 3. SIDEBAR CANDIDATE INPUTS
# -------------------------------------------------------------------
st.sidebar.header("📋 Candidate Input Profile")
name = st.sidebar.text_input("Candidate Name", "Rahul Verma")
experience = st.sidebar.number_input("Years of Experience", min_value=1, max_value=30, value=8)
role = st.sidebar.text_input("Role Requested", "Senior Specialist / Lead")
expected_ctc = st.sidebar.number_input("Expected Fixed CTC (in ₹ LPA)", min_value=3.0, max_value=100.0, value=20.0, step=0.5)
remote_req = st.sidebar.selectbox("Work Model Request", ["On-site (Plant / Office / Site)", "Hybrid Work", "100% Remote / WFH"])
bonus_req = st.sidebar.number_input("Requested Joining Bonus (in ₹ INR)", min_value=0, max_value=1000000, value=150000, step=25000)

run_button = st.sidebar.button("🚀 Evaluate Multi-Industry Offer")

if not groq_api_key:
    st.warning("👈 Pehle sidebar me apni **Groq API Key** enter karein taaki execution start ho sake.")
else:
    st.success(f"✅ Groq API Key Configured! Active Sector: **{industry}**")

# -------------------------------------------------------------------
# 4. SEQUENTIAL AGENT PIPELINE EXECUTION
# -------------------------------------------------------------------
if run_button:
    if not groq_api_key:
        st.error("⚠️ Please enter a valid Groq API Key in the sidebar before running.")
    else:
        with st.spinner(f"Evaluating Candidate for {industry} Sector..."):
            
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="openai/gpt-oss-120b",
                temperature=0.1
            )

            # --- AGENT 1: Industry Technical Recruiter ---
            recruiter_prompt = ChatPromptTemplate.from_messages([
                ("system", f"You are a Senior Technical Recruiter specializing in the **{industry}** sector in India. Evaluate candidate experience, role requested, and expected CTC (in ₹ LPA). Assign a seniority level (Level 1 to Level 5) with clear sector-specific justification."),
                ("human", "Candidate Details: Name: {name}, Experience: {exp} years, Requested Role: {role}, Expected CTC: ₹{ctc} LPA")
            ])
            recruiter_chain = recruiter_prompt | llm
            eval_result = clean_output(recruiter_chain.invoke({
                "name": name,
                "exp": experience,
                "role": role,
                "ctc": expected_ctc
            }).content)

            st.subheader(f"Step 1: 🎯 Recruiter Evaluation ({industry})")
            st.info(eval_result)

            # --- AGENT 2: Industry Policy Specialist (RAG Engine) ---
            policy_context = query_hr_policy(f"{role} work model joining bonus CTC allowance policy")
            
            policy_prompt = ChatPromptTemplate.from_messages([
                ("system", f"You are an HR Policy Specialist for the **{industry}** industry. Audit candidate terms against retrieved guidelines.\n\nRetrieved Sector Guidelines:\n{{context}}\n\nSTRICT FORMATTING RULES: Do not use HTML tags like <br>. Use clean Markdown bullets and tables. Ensure correct spacing for Indian Currency numbers (₹)."),
                ("human", "Recruiter Evaluation Output:\n{eval_output}\n\nCandidate Requests:\nExpected Fixed CTC: ₹{ctc} LPA\nWork Model Request: {remote}\nJoining Bonus Request: ₹{bonus:,}")
            ])
            policy_chain = policy_prompt | llm
            policy_result = clean_output(policy_chain.invoke({
                "context": policy_context,
                "eval_output": eval_result,
                "ctc": expected_ctc,
                "remote": remote_req,
                "bonus": bonus_req
            }).content)

            st.subheader("Step 2: ⚖️ HR Policy & Sector Compliance Audit")
            st.warning(policy_result)

            # --- AGENT 3: Compensation & Offer Strategist ---
            strategist_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an Executive Compensation Strategist in India specializing in **{industry}**. Synthesize candidate evaluations and HR policy audit reports into a final Hiring Offer Decision Package.

Provide a complete Indian CTC Breakdown table:
- Base Fixed CTC (₹ LPA)
- Basic Salary (50% of CTC)
- HRA (50% of Basic for Metro)
- Sector-Specific Allowances (Shift/Plant/Transport/Site/Special Allowances)
- Employer PF Contribution (12% of Basic)
- Approved Joining Bonus (in ₹)
- Approved Work Model & Notice Period

STRICT FORMATTING RULES: Use standard Markdown tables and bullet points. Do NOT use HTML tags like <br>."""),
                ("human", "Recruiter Evaluation:\n{eval_output}\n\nPolicy Compliance Audit:\n{audit_output}\n\nCreate a final formatted hiring offer recommendation suited for {industry} industry standards.")
            ])
            strategist_chain = strategist_prompt | llm
            final_offer = clean_output(strategist_chain.invoke({
                "eval_output": eval_result,
                "audit_output": policy_result
            }).content)

            st.success("✅ Multi-Agent Pipeline Execution Complete!")
            st.subheader(f"📜 Final Executive Offer Decision Package ({industry})")
            st.markdown(final_offer)

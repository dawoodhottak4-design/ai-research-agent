import os
import streamlit as st
import litellm
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# --- Crucial Fix for Groq Prompt Caching Error ---
# Unsupported parameters (like cache_breakpoint) ko auto-drop karne ke liye
litellm.drop_params = True

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Single-Agent AI Researcher")
st.write("Powered by **CrewAI**, **Groq**, and **DuckDuckGo**.")

# --- API Key Management ---
groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key:", 
    type="password",
    value=st.secrets.get("GROQ_API_KEY", "") if "GROQ_API_KEY" in st.secrets else ""
)

# Model Selector Dropdown
selected_model = st.sidebar.selectbox(
    "Select LLM Model:",
    [
        "groq/llama-3.3-70b-versatile",
        "groq/llama3-80b-8192",
        "groq/mixtral-8x7b-32768"
    ],
    index=0
)

if not groq_api_key:
    st.info("💡 Please enter your Groq API Key in the sidebar or set it in Streamlit Secrets to continue.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# --- Custom CrewAI Tool Definition ---
ddg_search = DuckDuckGoSearchRun()

@tool("DuckDuckGo Web Search")
def web_search_tool(query: str) -> str:
    """Search the web for information using DuckDuckGo."""
    try:
        return ddg_search.run(query)
    except Exception as e:
        return f"Error during search: {e}"

# --- User Inputs ---
topic = st.text_input(
    "Research Topic:", 
    placeholder="e.g., Latest trends in Educational Technology"
)

# --- Agent & Task Execution ---
if st.button("Generate Research Report", type="primary"):
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        with st.spinner("Agent is searching the web and compiling the report..."):
            try:
                # Native CrewAI LLM Initialization
                

                # 1. Define Agent
                research_agent = Agent(
                    rol# app.py mein LLM initialization block ko update karein:
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.3
)e="Senior Research Analyst",
                    goal=f"Conduct thorough, up-to-date web research on '{topic}' and generate a comprehensive report.",
                    backstory=(
                        "You are an expert analyst known for extracting precise insights from the web, "
                        "synthesizing complex facts, and formatting detailed, readable technical reports."
                    ),
                    tools=[web_search_tool],
                    llm=llm,
                    verbose=True,
                    allow_delegation=False
                )

                # 2. Define Task
                research_task = Task(
                    description=(
                        f"1. Search the web for recent and accurate details about: {topic}.\n"
                        f"2. Synthesize key trends, facts, statistics, and main considerations.\n"
                        f"3. Organize findings into a professional research report."
                    ),
                    expected_output=(
                        "A markdown-formatted report containing:\n"
                        "- Executive Summary\n"
                        "- Detailed Key Findings & Analysis\n"
                        "- Future Outlook / Implications\n"
                        "- References/Sources Summary"
                    ),
                    agent=research_agent
                )

                # 3. Create Crew
                crew = Crew(
                    agents=[research_agent],
                    tasks=[research_task],
                    process=Process.sequential
                )

                # 4. Kickoff Workflow
                result = crew.kickoff()

                # 5. Display Output
                st.success("Research Complete!")
                st.markdown("### 📋 Final Research Report")
                st.markdown(str(result))

                st.download_button(
                    label="📥 Download Report (.md)",
                    data=str(result),
                    file_name=f"{topic.lower().replace(' ', '_')}_report.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"An error occurred during execution: {e}")

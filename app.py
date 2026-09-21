import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Single-Agent AI Researcher")
st.write("Powered by **CrewAI**, **Groq (openai/gpt-oss-120b)**, and **DuckDuckGo**.")

# --- API Key Management ---
groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key:", 
    type="password",
    value=st.secrets.get("GROQ_API_KEY", "") if "GROQ_API_KEY" in st.secrets else ""
)

if not groq_api_key:
    st.info("💡 Please enter your Groq API Key in the sidebar or set it in Streamlit Secrets to continue.")
    st.stop()

# --- Initialize LLM & Tools ---
try:
    llm = ChatGroq(
        temperature=0.3,
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-120b"
    )
    search_tool = DuckDuckGoSearchRun()
except Exception as e:
    st.error(f"Error initializing services: {e}")
    st.stop()

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
                # 1. Define Agent
                research_agent = Agent(
                    role="Senior Research Analyst",
                    goal=f"Conduct thorough, up-to-date web research on '{topic}' and generate a comprehensive report.",
                    backstory=(
                        "You are an expert analyst known for extracting precise insights from the web, "
                        "synthesizing complex facts, and formatting detailed, readable technical reports."
                    ),
                    tools=[search_tool],
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

                # 4. Kickoff
                result = crew.kickoff()

                # 5. Output
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

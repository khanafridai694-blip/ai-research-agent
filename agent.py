import os

from crewai import LLM
from crewai_tools import SerperDevTool


def create_research_agent():
    """
    Creates our single research agent.
    """

    search_tool = SerperDevTool()

    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    researcher = Agent(
        role="AI Researcher",

        goal=(
            "Research the user's question carefully and produce "
            "an accurate, well-structured and evidence-based report."
        ),

        backstory=(
            "You are a professional research analyst. "
            "You investigate questions using reliable web sources, "
            "compare information from multiple sources, identify "
            "important facts, and clearly explain your findings."
        ),

        tools=[search_tool],

        verbose=True,

        allow_delegation=False
    )

    return researcher


def run_research(question):
    """
    Runs the research process for a user's question.
    """

    researcher = create_research_agent()

    research_task = Task(
        description=f"""
        Research the following question:

        {question}

        Follow these steps:

        1. Understand the research question.
        2. Break it into important subtopics.
        3. Search the web for relevant information.
        4. Prefer reliable and authoritative sources.
        5. Compare information from multiple sources.
        6. Identify important facts and findings.
        7. Mention uncertainty when information is unclear.
        8. Do not invent facts or sources.
        9. Create a clear and structured final report.
        10. Include the sources used in the research.

        The final report should contain:

        - Executive Summary
        - Introduction
        - Key Findings
        - Detailed Analysis
        - Important Considerations
        - Conclusion
        - Sources

        User's research question:
        {question}
        """,

        expected_output="""
        A detailed research report written in clear language.

        The report should contain:
        1. Executive Summary
        2. Introduction
        3. Key Findings
        4. Detailed Analysis
        5. Important Considerations
        6. Conclusion
        7. Sources
        """,

        agent=researcher
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()

    return result

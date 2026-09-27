import os

from crewai import Agent, Task, Crew, Process, LLM


def create_research_agent():

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
    )

    researcher = Agent(
        role="Professional Research Analyst",

        goal=(
            "Research the user's question carefully and produce "
            "an accurate, clear, well-structured answer."
        ),

        backstory=(
            "You are a professional research analyst. "
            "You analyze questions carefully and provide "
            "clear, factual and well-organized information."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher


def run_research(question):

    researcher = create_research_agent()

    research_task = Task(
        description=f"""
        Analyze the following research question:

        {question}

        Provide a detailed and useful answer.

        Do not invent facts, statistics, quotations, or sources.

        Structure the answer as:

        ## Executive Summary

        ## Introduction

        ## Key Findings

        ## Detailed Analysis

        ## Important Considerations

        ## Conclusion

        Research question:

        {question}
        """,

        expected_output="""
        A clear, detailed and well-structured research report
        containing:

        1. Executive Summary
        2. Introduction
        3. Key Findings
        4. Detailed Analysis
        5. Important Considerations
        6. Conclusion
        """,

        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    return result
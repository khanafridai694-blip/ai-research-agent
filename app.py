
import os
import streamlit as st

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]

from agent import run_research



# -------------------------
# Page configuration
# -------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide"
)


# -------------------------
# Title
# -------------------------

st.title("🔎 AI Research Assistant")

st.write(
    "Ask a research question and the AI agent will "
    "search the web and create a structured research report."
)


# -------------------------
# Sidebar
# -------------------------

with st.sidebar:

    st.header("Research Settings")

    st.write(
        "This version uses one CrewAI research agent "
        "with web search."
    )

    st.divider()

    st.info(
        "The research agent may take some time because "
        "it searches and analyzes multiple sources."
    )


# -------------------------
# Research question
# -------------------------

question = st.text_area(
    "What would you like me to research?",
    placeholder=(
        "Example: What are the major applications of "
        "artificial intelligence in healthcare?"
    ),
    height=150
)


# -------------------------
# Research button
# -------------------------

if st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a research question first."
        )

    else:

        st.info(
            "🔎 Research started. Please wait..."
        )

        try:

            with st.spinner(
                "The AI researcher is working..."
            ):

                result = run_research(question)

            st.success(
                "Research completed!"
            )

            st.divider()

            st.subheader(
                "📄 Research Report"
            )

            st.markdown(
                str(result)
            )

        except Exception as e:

            st.error(
                "Something went wrong."
            )

            st.exception(e)

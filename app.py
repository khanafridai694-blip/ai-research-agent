import os
import streamlit as st

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide",
)

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from agent import run_research


st.title("🔎 AI Research Assistant")

st.write(
    "Ask a question and the AI agent will create "
    "a structured research report."
)


with st.sidebar:
    st.header("⚙️ Research Settings")

    st.write("This application uses:")
    st.write("• CrewAI")
    st.write("• Groq")
    st.write("• Streamlit")

    st.divider()

    st.info(
        "The AI agent may take some time to analyze your question."
    )


question = st.text_area(
    "What would you like me to research?",

    placeholder=(
        "Example: What are the major applications "
        "of generative AI in software development?"
    ),

    height=150,
)


if st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
):

    if not question.strip():

        st.warning(
            "Please enter a research question first."
        )

    else:

        try:

            with st.spinner(
                "🔎 Researching... Please wait..."
            ):

                result = run_research(question)

            st.success("✅ Research completed!")

            st.divider()

            st.subheader("📄 Research Report")

            st.markdown(str(result))

        except Exception as e:

            st.error("❌ Something went wrong.")

            st.exception(e)
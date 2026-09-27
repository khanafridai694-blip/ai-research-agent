
"""
Groq + DuckDuckGo Search Agent
------------------------------
A single-agent Streamlit chat app where a Groq-hosted LLM can decide,
on its own, to search the web via DuckDuckGo before answering.

No search API key is required. Only a Groq API key is needed.
"""

import json
import time

import streamlit as st
from groq import Groq, APIStatusError, APIConnectionError, RateLimitError
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException


# =============================================================================
# 1. PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Groq Search Agent",
    page_icon="🔎",
    layout="centered",
)

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_SEARCH_RESULTS = 5
MAX_AGENT_STEPS = 4  # safety cap on tool-call loops


# =============================================================================
# 2. THE WEB SEARCH TOOL (DuckDuckGo, no API key needed)
# =============================================================================
def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """
    Perform a DuckDuckGo web search and return a formatted string of results.
    This is the function the LLM will "call" as a tool.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except DuckDuckGoSearchException as e:
        return f"SEARCH_ERROR: DuckDuckGo search failed ({e})."
    except Exception as e:
        return f"SEARCH_ERROR: Unexpected error during search ({e})."

    if not results:
        return "SEARCH_ERROR: No results found for this query. Try a different query."

    formatted = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        body = r.get("body", "No description")
        href = r.get("href", "No URL")
        formatted.append(f"[{i}] {title}\nURL: {href}\nSnippet: {body}")

    return "\n\n".join(formatted)


# Tool schema Groq needs to know when/how to call web_search
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web via DuckDuckGo for current, real-time, "
                "or factual information (news, prices, recent events, "
                "documentation, etc). Use this whenever the user's question "
                "needs up-to-date or specific information you are not "
                "certain about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to send to DuckDuckGo.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

AVAILABLE_FUNCTIONS = {"web_search": web_search}

SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to a `web_search` tool "
    "that queries DuckDuckGo. "
    "Use the tool whenever a question depends on current events, facts you "
    "are unsure of, or anything time-sensitive. "
    "You may call the tool multiple times with refined queries if needed. "
    "Once you have enough information, answer the user clearly and cite "
    "sources by URL where relevant. "
    "If search results are irrelevant or empty, say so honestly instead of "
    "making up an answer."
)


# =============================================================================
# 3. SIDEBAR — API KEY CONFIGURATION
# =============================================================================
def get_groq_api_key() -> str:
    """
    Resolve the Groq API key from (in priority order):
    1. Streamlit secrets (st.secrets) — used when deployed
    2. Sidebar text input — used for local/manual testing
    """
    secret_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

    with st.sidebar:
        st.header("⚙️ Configuration")
        if secret_key:
            st.success("Groq API key loaded from secrets.", icon="✅")
            return secret_key

        input_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at https://console.groq.com/keys",
        )
        st.caption(
            "🔒 Your key is only used for this session and is never stored."
        )
        st.divider()
        st.markdown(
            "**About this app**\n\n"
            "This agent uses **Groq** for fast LLM inference and "
            "**DuckDuckGo** for free, keyless web search. "
            "No search API key is required."
        )
        return input_key


# =============================================================================
# 4. AGENT LOOP
# =============================================================================
def run_agent(client: Groq, messages: list, status_box) -> str:
    """
    Run the tool-calling agent loop:
    1. Send conversation to Groq.
    2. If the model requests a tool call, execute it (DuckDuckGo search),
       append the result, and loop back.
    3. Stop when the model returns a plain text answer, or after
       MAX_AGENT_STEPS iterations as a safety net.
    """
    working_messages = messages.copy()

    for step in range(MAX_AGENT_STEPS):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=working_messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.3,
            )
        except RateLimitError:
            return (
                "⚠️ Groq rate limit reached. Please wait a moment and try "
                "again, or check your plan limits at console.groq.com."
            )
        except APIConnectionError:
            return (
                "⚠️ Could not connect to Groq's API. Please check your "
                "internet connection and try again."
            )
        except APIStatusError as e:
            return f"⚠️ Groq API returned an error (status {e.status_code}): {e.message}"
        except Exception as e:
            return f"⚠️ Unexpected error while contacting Groq: {e}"

        choice = response.choices[0]
        msg = choice.message

        # Case A: model wants to call one or more tools
        if msg.tool_calls:
            # Record the assistant's tool-call request in the conversation
            working_messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                func_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                query = args.get("query", "")
                status_box.write(f"🔍 Searching DuckDuckGo for: **{query}**")

                func = AVAILABLE_FUNCTIONS.get(func_name)
                if func is None:
                    tool_result = f"ERROR: Unknown tool '{func_name}'."
                else:
                    tool_result = func(**args)

                if tool_result.startswith("SEARCH_ERROR"):
                    status_box.write(f"⚠️ {tool_result}")
                else:
                    status_box.write("✅ Results retrieved, analyzing...")

                # Feed the tool result back into the conversation
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": func_name,
                        "content": tool_result,
                    }
                )
            # Loop again so the model can read tool results and respond
            continue

        # Case B: model returned a final plain-text answer
        if msg.content:
            return msg.content

        # Fallback (shouldn't normally happen)
        return "I wasn't able to generate a response. Please try rephrasing your question."

    return (
        "I searched several times but couldn't reach a confident final "
        "answer. Please try rephrasing your question or being more specific."
    )


# =============================================================================
# 5. MAIN UI
# =============================================================================
def main():
    st.title("🔎 Groq Search Agent")
    st.caption(
        "Ask anything. This agent uses **Groq** (Llama 3.3 70B) for reasoning "
        "and **DuckDuckGo** for free, real-time web search."
    )

    api_key = get_groq_api_key()

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Render past messages (skip the system prompt)
    for message in st.session_state.messages:
        if message["role"] in ("user", "assistant") and message.get("content"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input box
    user_prompt = st.chat_input("Ask a question...")

    if user_prompt:
        if not api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar to continue.")
            st.stop()

        # Show and store the user's message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Run the agent and show its reasoning steps live
        with st.chat_message("assistant"):
            with st.status("🤖 Agent is thinking...", expanded=True) as status_box:
                try:
                    client = Groq(api_key=api_key)
                except Exception as e:
                    st.error(f"⚠️ Could not initialize Groq client: {e}")
                    st.stop()

                start_time = time.time()
                answer = run_agent(client, st.session_state.messages, status_box)
                elapsed = time.time() - start_time
                status_box.update(
                    label=f"✅ Done in {elapsed:.1f}s", state="complete", expanded=False
                )

            st.markdown(answer)

        # Save only the clean final answer to history
        # (tool-call plumbing is not persisted across turns to keep context light)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Sidebar: clear chat button
    with st.sidebar:
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.rerun()


if __name__ == "__main__":
    main()

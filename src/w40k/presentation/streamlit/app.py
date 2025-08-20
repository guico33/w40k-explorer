"""Streamlit app for the W40K Knowledge Base - New Architecture.

Supports running as a script by falling back to add `<repo>/src` to sys.path
when the `w40k` package is not importable.
"""

import sys
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from w40k.config.factory import create_answer_service, validate_environment
    from w40k.core.models import QueryResult
    from w40k.presentation.streamlit.utils import (
        format_sources_with_sequential_numbering,
    )
except Exception:
    src_dir = Path(__file__).resolve().parents[4]  # <repo>/src
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from w40k.config.factory import create_answer_service, validate_environment
    from w40k.core.models import QueryResult
    from w40k.presentation.streamlit.utils import (
        format_sources_with_sequential_numbering,
    )


def initialize_app():
    """Initialize the Streamlit app."""
    st.set_page_config(
        page_title="Warhammer 40K Knowledge Base",
        page_icon="⚔️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("⚔️ Warhammer 40K Knowledge Base")
    st.markdown("*Ask any question about the grim darkness of the far future...*")

    # Validate environment
    is_valid, error_msg = validate_environment()
    if not is_valid:
        st.error(f"❌ **Environment Error**: {error_msg}")
        st.info(
            "Please check your `.env` file and ensure all required variables are set."
        )
        st.stop()


@st.cache_resource
def get_answer_service():
    """Initialize and cache the answer service."""
    try:
        with st.spinner("🔄 Initializing knowledge base..."):
            answer_service, stats = create_answer_service()

        # Unified success message (no SQLite distinction)
        st.success("✅ Knowledge base initialized")

        # Show stats in sidebar
        with st.sidebar:
            st.header("📊 System Status")
            st.metric("Vector Database", f"{stats['chunks_count']} chunks")
            # Optional provider if present
            if "provider" in stats:
                st.info(f"**Provider**: {str(stats['provider']).title()}")
            st.info(f"**Model**: {stats['model']}")
            st.info(f"**Connection**: {stats['connection_info']}")

        return answer_service

    except Exception as e:
        st.error(f"❌ **Initialization Error**: {str(e)}")
        st.info("Please check your vector service connection and environment variables.")
        st.stop()


def initialize_session_state():
    """Initialize session state for chat history."""
    if "history" not in st.session_state:
        st.session_state.history = []


def display_message(role: str, content: str, result: Optional[QueryResult] = None):
    """Display a chat message with optional query result metadata."""
    with st.chat_message(role):
        if result and role == "assistant":
            _render_assistant_output(content, result)
        else:
            st.markdown(content)


def _render_assistant_output(answer_text: str, result: QueryResult) -> None:
    """Render assistant answer, sources (collapsible), metadata, and errors."""
    # Remap citations and display formatted answer
    remapped_answer, sources_text = format_sources_with_sequential_numbering(
        answer_text, result.citations
    )
    st.markdown(remapped_answer)

    # Display sources inline (as before)
    st.markdown("**📚 Sources:**")
    if not result.citations and not getattr(result, "citations_used", []):
        st.info("Sources unavailable: structured output missing or invalid JSON.")
    else:
        st.markdown(sources_text)

    # Collapsible: show full passages matching the same numbering as sources
    if getattr(result, "context_items", None) and result.citations:
        # Build mapping original context id -> sequential number based on sources order
        id_to_seq = {
            c.get("id"): i + 1 for i, c in enumerate(result.citations) if c.get("id") is not None
        }
        if id_to_seq:
            with st.expander("Show full passages"):
                for i, c in enumerate(result.citations):
                    cid = c.get("id")
                    if isinstance(cid, int) and 0 <= cid < len(result.context_items):
                        ctx = result.context_items[cid]
                        seq = id_to_seq.get(cid, i + 1)
                        st.markdown(
                            f"**[{seq}] {ctx['article']}** › {ctx['section']}\n\n{ctx['text']}\n\n"
                        )
                        if ctx.get("url"):
                            st.markdown(f"[Link]({ctx['url']})")

    # Display metadata
    confidence_color = (
        "green"
        if result.confidence > 0.7
        else "orange" if result.confidence > 0.5 else "red"
    )
    metadata_text = (
        f"*Confidence: <span style='color:{confidence_color}'>{result.confidence:.2f}</span> • "
        f"Query time: {result.query_time_ms}ms*"
    )
    st.markdown(metadata_text, unsafe_allow_html=True)

    # Display error if any
    if result.error:
        st.warning(f"⚠️ **Note**: {result.error}")


def main():
    """Main Streamlit app function."""
    # Initialize app
    initialize_app()

    # Initialize answer service (cached)
    answer_service = get_answer_service()

    # Initialize session state
    initialize_session_state()

    # Display chat history
    for message in st.session_state.history:
        display_message(message["role"], message["content"], message.get("result"))

    # Chat input
    if prompt := st.chat_input("Ask about Warhammer 40K lore..."):
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": prompt})

        # Display user message
        display_message("user", prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching the archives..."):
                try:
                    result = answer_service.answer_query(prompt)

                    _render_assistant_output(result.answer, result)

                    # Add assistant message to history (store remapped answer)
                    st.session_state.history.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "result": result,
                        }
                    )

                except Exception as e:
                    error_msg = f"❌ **Error processing query**: {str(e)}"
                    st.error(error_msg)

                    # Add error message to history
                    st.session_state.history.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    main()

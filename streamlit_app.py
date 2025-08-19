"""Warhammer 40K Knowledge Base - Streamlit Chat Interface (New Architecture)."""

from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Rely on installed package `w40k` (editable install or PYTHONPATH=src)

# Import from new architecture
from w40k.config.factory import create_answer_service, validate_environment
from w40k.config.settings import get_settings
from w40k.core.models import QueryResult
from w40k.presentation.streamlit.utils import format_sources_with_sequential_numbering


def initialize_app():
    """Initialize the Streamlit app with page config and environment validation."""
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
            "Please check your environment variables and ensure all required variables are set."
        )
        st.stop()


@st.cache_resource
def get_answer_service():
    """Initialize and cache the answer service."""
    try:
        with st.spinner("🔄 Initializing knowledge base..."):
            # Auto-detect SQLite availability via Settings
            settings = get_settings()
            use_sqlite = settings.db_exists()
            answer_service, stats = create_answer_service(use_sqlite=use_sqlite)

        # Display initialization success
        if use_sqlite:
            st.success("✅ Knowledge base initialized successfully!")
        else:
            st.success("✅ Knowledge base initialized (Qdrant-only mode)!")

        # Show stats in sidebar
        with st.sidebar:
            st.header("📊 System Status")
            st.metric("Vector Database", f"{stats['chunks_count']} chunks")
            st.metric("Coverage", f"{stats['coverage_percentage']:.1f}%")
            st.info(f"**Model**: {stats['model']}")
            st.info(f"**Connection**: {stats['connection_info']}")
            if not use_sqlite:
                st.info("🔧 **Mode**: Qdrant-only (no SQLite)")

        return answer_service

    except Exception as e:
        st.error(f"❌ **Initialization Error**: {str(e)}")
        st.info("Please check your Qdrant connection and environment variables.")
        st.stop()


def initialize_session_state():
    """Initialize session state for chat history."""
    if "history" not in st.session_state:
        st.session_state.history = []


def display_message(role: str, content: str, result: Optional[QueryResult] = None):
    """Display a chat message with optional query result metadata."""
    with st.chat_message(role):
        if result and role == "assistant":
            # Remap citations and display formatted answer
            remapped_answer, sources_text = format_sources_with_sequential_numbering(
                content, result.citations
            )
            st.markdown(remapped_answer)

            # Display sources
            st.markdown("**📚 Sources:**")
            st.markdown(sources_text)

            # Display metadata
            confidence_color = (
                "green"
                if result.confidence > 0.7
                else "orange" if result.confidence > 0.5 else "red"
            )
            metadata_text = f"*Confidence: <span style='color:{confidence_color}'>{result.confidence:.2f}</span> • Query time: {result.query_time_ms}ms*"
            st.markdown(metadata_text, unsafe_allow_html=True)

            # Display error if any
            if result.error:
                st.warning(f"⚠️ **Note**: {result.error}")
        else:
            st.markdown(content)


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

                    # Remap citations and display formatted answer
                    remapped_answer, sources_text = (
                        format_sources_with_sequential_numbering(
                            result.answer, result.citations
                        )
                    )
                    st.markdown(remapped_answer)

                    # Display sources
                    st.markdown("**📚 Sources:**")
                    st.markdown(sources_text)

                    # Display metadata
                    confidence_color = (
                        "green"
                        if result.confidence > 0.7
                        else "orange" if result.confidence > 0.5 else "red"
                    )
                    metadata_text = f"*Confidence: <span style='color:{confidence_color}'>{result.confidence:.2f}</span> • Query time: {result.query_time_ms}ms*"
                    st.markdown(metadata_text, unsafe_allow_html=True)

                    # Display error if any
                    if result.error:
                        st.warning(f"⚠️ **Note**: {result.error}")

                    # Add assistant message to history (store remapped answer)
                    st.session_state.history.append(
                        {
                            "role": "assistant",
                            "content": remapped_answer,
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

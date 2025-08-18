"""Warhammer 40K Knowledge Base - Streamlit Chat Interface."""

import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# Ensure project root is on sys.path and use absolute package imports
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import our modules using the package root
from src.engine.bootstrap import validate_environment, setup_query_engine
from src.engine.types import QueryResult


def initialize_app():
    """Initialize the Streamlit app with page config and environment validation."""
    st.set_page_config(
        page_title="Warhammer 40K Knowledge Base",
        page_icon="⚔️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("⚔️ Warhammer 40K Knowledge Base")
    st.markdown("*Ask any question about the grim darkness of the far future...*")
    
    # Validate environment
    is_valid, error_msg = validate_environment()
    if not is_valid:
        st.error(f"❌ **Environment Error**: {error_msg}")
        st.info("Please check your `.env` file and ensure all required variables are set.")
        st.stop()


@st.cache_resource
def get_query_engine():
    """Initialize and cache the query engine."""
    try:
        with st.spinner("🔄 Initializing query engine..."):
            engine, stats = setup_query_engine()
        
        # Display initialization success
        st.success("✅ Query engine initialized successfully!")
        
        # Show stats in sidebar
        with st.sidebar:
            st.header("📊 System Status")
            st.metric("Vector Database", f"{stats['chunks_count']} chunks")
            st.metric("Coverage", f"{stats['coverage_percentage']:.1f}%")
            st.info(f"**Model**: {stats['model']}")
            st.info(f"**Connection**: {stats['connection_info']}")
        
        return engine
    
    except Exception as e:
        st.error(f"❌ **Initialization Error**: {str(e)}")
        st.info("Please ensure your database exists and all services are running.")
        st.stop()


def initialize_session_state():
    """Initialize session state for chat history."""
    if "history" not in st.session_state:
        st.session_state.history = []


def format_sources(citations: List[Dict]) -> str:
    """Format citation sources for display."""
    if not citations:
        return "No sources available"
    
    formatted_sources = []
    for citation in citations:
        title = citation.get("title", "Unknown")
        section = citation.get("section", "")
        url = citation.get("url", "")
        
        if section:
            source_text = f"**{title}** › {section}"
        else:
            source_text = f"**{title}**"
        
        if url:
            source_text = f"[{source_text}]({url})"
        
        formatted_sources.append(f"• {source_text}")
    
    return "\n".join(formatted_sources)


def display_message(role: str, content: str, result: Optional[QueryResult] = None):
    """Display a chat message with optional query result metadata."""
    with st.chat_message(role):
        st.markdown(content)
        
        if result and role == "assistant":
            # Display sources
            st.markdown("**📚 Sources:**")
            sources_text = format_sources(result.citations)
            st.markdown(sources_text)
            
            # Display metadata
            confidence_color = "green" if result.confidence > 0.7 else "orange" if result.confidence > 0.5 else "red"
            metadata_text = f"*Confidence: <span style='color:{confidence_color}'>{result.confidence:.2f}</span> • Query time: {result.query_time_ms}ms*"
            st.markdown(metadata_text, unsafe_allow_html=True)
            
            # Display error if any
            if result.error:
                st.warning(f"⚠️ **Note**: {result.error}")


def main():
    """Main Streamlit app function."""
    # Initialize app
    initialize_app()
    
    # Initialize query engine (cached)
    engine = get_query_engine()
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat history
    for message in st.session_state.history:
        display_message(
            message["role"], 
            message["content"], 
            message.get("result")
        )
    
    # Chat input
    if prompt := st.chat_input("Ask about Warhammer 40K lore..."):
        # Add user message to history
        st.session_state.history.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        display_message("user", prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching the archives..."):
                try:
                    result = engine.query(prompt)
                    
                    # Display answer
                    st.markdown(result.answer)
                    
                    # Display sources
                    st.markdown("**📚 Sources:**")
                    sources_text = format_sources(result.citations)
                    st.markdown(sources_text)
                    
                    # Display metadata
                    confidence_color = "green" if result.confidence > 0.7 else "orange" if result.confidence > 0.5 else "red"
                    metadata_text = f"*Confidence: <span style='color:{confidence_color}'>{result.confidence:.2f}</span> • Query time: {result.query_time_ms}ms*"
                    st.markdown(metadata_text, unsafe_allow_html=True)
                    
                    # Display error if any
                    if result.error:
                        st.warning(f"⚠️ **Note**: {result.error}")
                    
                    # Add assistant message to history
                    st.session_state.history.append({
                        "role": "assistant",
                        "content": result.answer,
                        "result": result
                    })
                    
                except Exception as e:
                    error_msg = f"❌ **Error processing query**: {str(e)}"
                    st.error(error_msg)
                    
                    # Add error message to history
                    st.session_state.history.append({
                        "role": "assistant", 
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()

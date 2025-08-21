"""Streamlit app for W40K Explorer - New Architecture.

Supports running as a script by falling back to add `<repo>/src` to sys.path
when the `w40k` package is not importable.
"""

import base64
import sys
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from streamlit.components.v1 import html as st_html

# Load environment variables
load_dotenv()

try:
    from w40k.config.factory import create_answer_service, validate_environment
    from w40k.core.models import QueryResult
    from w40k.presentation.streamlit.utils import (
        format_sources_with_sequential_numbering,
        style_citation_superscripts_with_links,
        build_seq_to_url_map,
        format_single_source,
    )
except Exception:
    src_dir = Path(__file__).resolve().parents[4]  # <repo>/src
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from w40k.config.factory import create_answer_service, validate_environment
    from w40k.core.models import QueryResult
    from w40k.presentation.streamlit.utils import (
        format_sources_with_sequential_numbering,
        style_citation_superscripts_with_links,
        build_seq_to_url_map,
        format_single_source,
    )


def get_base64_image(image_path):
    """Convert image to base64 string for CSS embedding."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None


def get_chat_avatar(role: str):
    """Get the appropriate avatar for chat messages."""
    current_dir = Path(__file__).parent
    if role == "user":
        avatar_path = current_dir / "assets" / "user.png"
    else:  # assistant
        avatar_path = current_dir / "assets" / "assistant.png"

    if avatar_path.exists():
        return str(avatar_path)
    return None


def get_themed_icon(icon_type: str, size: int = 20) -> str:
    """Get themed icon as base64 embedded HTML."""
    current_dir = Path(__file__).parent
    icon_mapping = {"gear": "gear.png", "seal": "seal.png", "book": "book.png"}

    if icon_type not in icon_mapping:
        return ""

    icon_path = current_dir / "assets" / icon_mapping[icon_type]
    if not icon_path.exists():
        return ""

    base64_icon = get_base64_image(icon_path)
    if base64_icon:
        return f'<img src="data:image/png;base64,{base64_icon}" width="{size}" height="{size}" style="vertical-align: middle; margin-right: 8px;">'
    return ""


def add_background_image():
    """Add the Warhammer 40K background image with CSS."""
    # Get the path to the background image
    current_dir = Path(__file__).parent
    image_path = current_dir / "assets" / "background.jpg"

    # Convert image to base64
    base64_image = get_base64_image(image_path)

    if base64_image:
        st.markdown(
            f"""
        <style>
        /* Import Gothic fonts */
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Uncial+Antiqua&family=Pirata+One&display=swap');
        
        /* CSS Variables for W40K Theme */
        :root {{
            --w40k-gold: #c9aa71;
            --w40k-gold-alpha-30: rgba(201, 170, 113, 0.3);
            --w40k-gold-alpha-60: rgba(201, 170, 113, 0.6);
            --w40k-dark-overlay: rgba(0, 0, 0, 0.3);
        }}
        
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Eliminate any global top spacing */
        html, body {{
            margin: 0 !important;
            padding: 0 !important;
        }}
        [data-testid="stAppViewContainer"] {{
            padding-top: 0 !important;
        }}
        section.main {{
            padding-top: 0 !important;
            margin-top: 0 !important;
        }}

        /* Remove Streamlit default top header space */
        [data-testid="stHeader"] {{
            display: none !important;
            height: 0 !important;
            visibility: hidden !important;
        }}

        /* Disable smooth scroll to reduce jumpiness on anchor updates */
        html {{
            scroll-behavior: auto !important;
        }}
        
        /* Add semi-transparent overlay for better readability */
        .stApp::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.4);
            z-index: -1;
            pointer-events: none;
        }}
        
        /* Gothic title styling - ONLY for the main title */
        .block-container > div:first-child h1 {{
            font-family: 'Cinzel', 'Uncial Antiqua', 'Times New Roman', serif !important;
            font-weight: 700 !important;
            font-size: 3.5rem !important;
            text-align: center !important;
            color: var(--w40k-gold) !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8), 
                         0 0 10px var(--w40k-gold-alpha-30) !important;
            margin-top: 0 !important;
            margin-bottom: 0.5rem !important;
            letter-spacing: 2px !important;
        }}
        
        /* Hide anchor links from headers */
        .stHeaderActionElements,
        [data-testid="stHeaderActionElements"] {{
            display: none !important;
        }}
        
        /* Also hide anchor links in expanders and other headers */
        .streamlit-expanderHeader [data-testid="stHeaderActionElements"],
        h1 [data-testid="stHeaderActionElements"],
        h2 [data-testid="stHeaderActionElements"],
        h3 [data-testid="stHeaderActionElements"] {{
            display: none !important;
        }}

        /* Ensure any residual header anchor elements don't capture clicks */
        .streamlit-expanderHeader a[href^="#"],
        h1 a[href^="#"],
        h2 a[href^="#"],
        h3 a[href^="#"] {{
            pointer-events: none !important;
        }}
        
        /* Enhance text readability and tighten top spacing */
        .main .block-container {{
            background: var(--w40k-dark-overlay);
            backdrop-filter: blur(2px);
            border-radius: 10px;
            padding: 0 2rem 2rem 2rem !important;
            margin-top: 0 !important;
        }}
        
        /* Remove any default top margin from the first block */
        .block-container > div:first-child {{
            margin-top: 0 !important;
        }}
        .block-container {{
            padding-top: 0 !important;
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            background: rgba(20, 20, 30, 0.9);
            backdrop-filter: blur(10px);
        }}
        
        /* Chat message containers */
        .stChatMessage {{
            background: rgba(40, 40, 50, 0.8);
            backdrop-filter: blur(5px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        /* Larger font for all chat messages */
        .stChatMessage p,
        .stChatMessage .stMarkdown p {{
            font-size: 1.15rem !important;
            line-height: 1.6 !important;
        }}

        /* Superscript citation markers (styled as links) */
        .citation-link,
        .citation-link:link,
        .citation-link:visited,
        .citation-link:hover,
        .citation-link:active {{
            text-decoration: none !important;
            border-bottom: none !important;
            color: var(--w40k-gold) !important;
        }}
        .citation-link .citation-sup {{
            text-decoration: none !important;
            border-bottom: none !important;
        }}
        .citation-sup {{
            vertical-align: super;
            font-size: 0.75em;
            line-height: 0;
            color: var(--w40k-gold) !important;
            font-weight: 700 !important;
            margin-left: 1px;
        }}

        /* Make all links inside chat messages gold (includes Sources and Passages) */
        .stChatMessage a,
        .stChatMessage a:link,
        .stChatMessage a:visited,
        .stChatMessage a:hover,
        .stChatMessage a:active {{
            color: var(--w40k-gold) !important;
        }}
        
        /* Mobile Responsiveness */
        @media (max-width: 768px) {{
            /* Smaller title on mobile */
            .block-container > div:first-child h1 {{
                font-size: 2.5rem !important;
                letter-spacing: 1px !important;
            }}
            
            /* Adjust container padding for mobile */
            .main .block-container {{
                padding: 0 1rem 1rem 1rem !important;
            }}
            
            /* Smaller chat message font on very small screens */
            .stChatMessage p,
            .stChatMessage .stMarkdown p {{
                font-size: 1rem !important;
            }}
        }}
        
        @media (max-width: 480px) {{
            /* Even smaller title on very small screens */
            .block-container > div:first-child h1 {{
                font-size: 2rem !important;
            }}
            
            /* Tighter padding on small screens */
            .main .block-container {{
                padding: 0 0.5rem 0.5rem 0.5rem !important;
            }}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


def disable_anchor_scroll_behavior():
    """Inject JS to neutralize hash-anchor clicks that can cause page jumps.

    Some Streamlit builds add anchor links to headings. Even when hidden, these can
    still update `location.hash` and trigger scroll jumps. This script prevents
    default for anchor clicks whose href starts with '#', and clears any existing
    hash on load.
    """
    st_html(
        """
        <script>
        (function() {
          function isHashLink(el) { return el && el.getAttribute && /^#/.test(el.getAttribute('href')||''); }
          document.addEventListener('click', function(ev){
            var a = ev.target.closest ? ev.target.closest('a') : null;
            if (isHashLink(a) && !(a.classList && a.classList.contains('citation-link'))) {
              ev.preventDefault();
              return false;
            }
          }, true);
          if (location.hash) {
            try { history.replaceState('', document.title, location.pathname + location.search); } catch(e) {}
          }
        })();
        </script>
        """,
        height=0,
    )


def initialize_app():
    """Initialize the Streamlit app."""
    st.set_page_config(
        page_title="Warhammer 40K Explorer",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Add the background image
    add_background_image()
    # Disable anchor scroll behavior to avoid jumpy expanders/headings
    disable_anchor_scroll_behavior()

    st.title("Warhammer 40K Explorer")
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
        seal_icon = get_themed_icon("seal", size=24)
        st.markdown(
            f"""
        <div style='padding: 0.75rem; background: var(--w40k-gold-alpha-30); border: 1px solid var(--w40k-gold-alpha-60); border-radius: 0.375rem; color: var(--w40k-gold); margin: 1rem 0; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);'>
            {seal_icon}<strong>Knowledge base initialized</strong>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Show stats in sidebar - collapsible and collapsed by default
        with st.sidebar:
            with st.expander("System Status", expanded=False):
                st.metric("Vector Database", f"{stats['chunks_count']} chunks")
                # Optional provider if present
                if "provider" in stats:
                    st.info(f"**Provider**: {str(stats['provider']).title()}")
                st.info(f"**Model**: {stats['model']}")
                st.info(f"**Connection** - {stats['connection_info']}")

        return answer_service

    except Exception as e:
        st.error(f"❌ **Initialization Error**: {str(e)}")
        st.info(
            "Please check your vector service connection and environment variables."
        )
        st.stop()


def initialize_session_state():
    """Initialize session state for chat history."""
    if "history" not in st.session_state:
        st.session_state.history = []


def display_message(role: str, content: str, result: Optional[QueryResult] = None):
    """Display a chat message with optional query result metadata."""
    avatar = get_chat_avatar(role)
    with st.chat_message(role, avatar=avatar):
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
    # Build mapping from sequential number -> external URL
    seq_to_url = build_seq_to_url_map(answer_text, result.citations)
    styled_answer = style_citation_superscripts_with_links(remapped_answer, seq_to_url)
    st.markdown(styled_answer, unsafe_allow_html=True)

    # Display sources inline (as before)
    book_icon = get_themed_icon("book", size=24)
    st.markdown(f"**{book_icon}Sources:**", unsafe_allow_html=True)
    if not result.citations and not getattr(result, "citations_used", []):
        st.info("Sources unavailable: structured output missing or invalid JSON.")
    else:
        st.markdown(sources_text, unsafe_allow_html=True)

    # Collapsible: show full passages matching the same numbering as sources
    if getattr(result, "context_items", None) and result.citations:
        # Build mapping original context id -> sequential number based on sources order
        id_to_seq = {
            c.get("id"): i + 1
            for i, c in enumerate(result.citations)
            if c.get("id") is not None
        }
        if id_to_seq:
            with st.expander("Show full passages"):
                for i, c in enumerate(result.citations):
                    cid = c.get("id")
                    if isinstance(cid, int) and 0 <= cid < len(result.context_items):
                        ctx = result.context_items[cid]
                        seq = id_to_seq.get(cid, i + 1)
                        # Format header like Sources (with linked title/section)
                        header_html = format_single_source(
                            {
                                "id": cid,
                                "title": c.get("title", ctx.get("article", "Unknown")),
                                "section": c.get("section", ctx.get("section", "")),
                                "url": c.get("url", ctx.get("url", "")),
                            },
                            seq,
                        )
                        st.markdown(header_html, unsafe_allow_html=True)
                        st.markdown(ctx.get("text", ""))

    # Display metadata
    metadata_text = (
        f"*Confidence: <span style='color:var(--w40k-gold); font-weight: bold;'>{result.confidence:.2f}</span> • "
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
        assistant_avatar = get_chat_avatar("assistant")
        with st.chat_message("assistant", avatar=assistant_avatar):
            with st.spinner("Searching the archives..."):
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

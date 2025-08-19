"""Streamlit Cloud compatibility shim.

This module preserves the historical entrypoint `src/ui/streamlit_app.py`
by delegating to the new Streamlit app located at
`w40k/presentation/streamlit/app.py`.
"""

from w40k.presentation.streamlit.app import main as _run

# Run immediately so Streamlit executes the app when this module is the main file
_run()


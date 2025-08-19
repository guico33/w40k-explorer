"""Presentation layer - UI and external interfaces."""

from .cli import W40KChatCLI, main as cli_main

__all__ = ["W40KChatCLI", "cli_main"]
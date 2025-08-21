#!/usr/bin/env python3
"""Interactive CLI chat interface for the Warhammer 40K knowledge base."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..config.factory import create_answer_service, validate_environment
from ..config.settings import Settings, get_settings
from ..core.models import QueryResult

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class W40KChatCLI:
    """Command-line interface for the W40K knowledge base."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize CLI with optional settings override."""
        self.settings = settings or get_settings()
        self.answer_service = None
        self.stats = None

    def initialize(
        self,
        initial_k: int = 60,
        min_score: Optional[float] = 0.2,
        max_context: int = 12,
        context_max_words: int = 200,
        expand_queries: int = 0,
        lower_threshold_on_empty: bool = True,
        active_only: bool = True,
        verbose: bool = False,
    ) -> bool:
        """Initialize the answer service with given parameters."""

        print("📚 Initializing W40K knowledge base")

        try:
            if verbose:
                print("  🔧 Setting up components...")

            self.answer_service, self.stats = create_answer_service(
                initial_k=initial_k,
                min_score=min_score,
                max_context=max_context,
                context_max_words=context_max_words,
                expand_queries=expand_queries,
                lower_threshold_on_empty=lower_threshold_on_empty,
                active_only=active_only,
                settings=self.settings,
            )

            print(f"✅ Knowledge base initialized")
            print(f"🤖 Model: {self.stats['model']}")
            print(f"🔌 Connection: {self.stats['connection_info']}")

            if self.stats["chunks_count"] != "unknown":
                print(f"📊 Vector database: {self.stats['chunks_count']} chunks")

            if self.stats.get("error"):
                print(f"⚠️  Warning: {self.stats['error']}")

            return True

        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            return False

    def display_result(self, result: QueryResult, verbose: bool = False):
        """Display query result in a formatted way."""

        print("\n" + "=" * 60)

        # Display answer
        print(
            f"📝 **Answer** (Confidence: {result.confidence:.2f}, {result.query_time_ms}ms):"
        )
        print(f"{result.answer}")

        # Display citations
        if result.citations:
            print(f"\n📚 **Sources** ({len(result.citations)} used):")
            for cite in result.citations:
                if cite.get("url"):
                    print(f"• {cite['title']} › {cite['section']} ({cite['url']})")
                else:
                    print(f"• {cite['title']} › {cite['section']}")
        else:
            print("\n📚 **Sources**: None")

        # Display confidence indicator
        if result.confidence < 0.5:
            print("\n⚠️  Low confidence answer - limited sources available")
        elif result.confidence < 0.7:
            print("\nℹ️  Moderate confidence - additional sources might help")

        # Display error if any
        if result.error:
            print(f"\n❌ **Error**: {result.error}")

        # Verbose debug info
        if verbose:
            print(f"\n🔍 **Debug Info**:")
            print(f"  - Sources processed: {result.sources_used}")
            print(f"  - Citations generated: {len(result.citations)}")
            print(f"  - Query time: {result.query_time_ms}ms")
            if hasattr(result, "citations_used"):
                print(f"  - Citation IDs used: {result.citations_used}")
            if result.error:
                print(f"  - Error: {result.error}")

        print("=" * 60)

    def run_single_query(self, question: str, verbose: bool = False):
        """Run a single query and display results."""
        if not self.answer_service:
            print("❌ Answer service not initialized")
            return False

        print(f"\n🔍 Processing query: {question}")
        print("🤔 Searching archives...")

        try:
            result = self.answer_service.answer_query(question)
            self.display_result(result, verbose)
            return True
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            return False

    def interactive_chat(self, verbose: bool = False):
        """Start interactive chat loop."""
        if not self.answer_service:
            print("❌ Answer service not initialized")
            return

        print("\n🤖 Warhammer 40K Explorer - Interactive Chat")
        print("💬 Type your question (or 'quit' to exit):\n")

        while True:
            try:
                # Get user input
                question = input("> ").strip()

                # Handle exit commands
                if question.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                # Handle empty input
                if not question:
                    continue

                # Handle help command
                if question.lower() in ["help", "?"]:
                    print("\n📋 Available commands:")
                    print("  - Ask any question about Warhammer 40K lore")
                    print("  - 'quit', 'exit', 'q' - Exit the chat")
                    print("  - 'help', '?' - Show this help message")
                    continue

                # Process the query
                print("🤔 Searching archives...")

                result = self.answer_service.answer_query(question)

                # Display results
                self.display_result(result, verbose)

                print(f"\n💬 Type your next question (or 'quit' to exit):")

            except KeyboardInterrupt:
                print("\n� Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()
                print("\n💬 Try another question:")


def main():
    """Main CLI entry point."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat interface for Warhammer 40K knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Interactive chat mode
  %(prog)s -q "Who is Horus?"           # Single query mode
  %(prog)s -v                           # Verbose output
        """,
    )

    # Note: Inference does not require SQLite; embeddings/ingestion scripts handle DB.
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose debug information"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="Initial number of chunks to retrieve (default: 60)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum similarity score threshold (omit to disable)",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=12,
        help="Maximum number of context chunks (default: 12)",
    )
    parser.add_argument(
        "--context-max-words",
        type=int,
        default=200,
        help="Max words per context chunk (default: 200)",
    )
    parser.add_argument(
        "--expand-queries",
        type=int,
        default=0,
        help="Number of query expansions to generate (0 disables)",
    )
    parser.add_argument(
        "--no-relax-threshold",
        action="store_true",
        help="Do not relax threshold when no hits are found",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include inactive chunks in vector search",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Run a single query and exit (non-interactive mode)",
    )

    args = parser.parse_args()

    print("🚀 Starting W40K Explorer CLI")
    print("=" * 40)

    # Validate environment
    is_valid, error_msg = validate_environment()
    if not is_valid:
        print(f"❌ Environment validation failed: {error_msg}")
        sys.exit(1)

    print("✅ Environment variables validated")

    # Setup CLI with verbose logging if requested
    if args.verbose:
        logging.getLogger("src.w40k").setLevel(logging.DEBUG)
        logging.getLogger("w40k").setLevel(logging.DEBUG)

    # Initialize CLI
    cli = W40KChatCLI()

    success = cli.initialize(
        initial_k=args.k,
        min_score=args.min_score,
        max_context=args.max_context,
        context_max_words=args.context_max_words,
        expand_queries=args.expand_queries,
        lower_threshold_on_empty=not args.no_relax_threshold,
        active_only=not args.include_inactive,
        verbose=args.verbose,
    )

    if not success:
        sys.exit(1)

    # Handle direct query mode or interactive chat
    if args.query:
        # Non-interactive mode: run single query and exit
        success = cli.run_single_query(args.query, args.verbose)
        sys.exit(0 if success else 1)
    else:
        # Interactive chat mode
        try:
            cli.interactive_chat(args.verbose)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()

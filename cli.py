#!/usr/bin/env python3
"""
CLI Interface — Interactive Document Q&A Bot

Provides a beautiful command-line interface for asking questions
against the indexed document collection. Displays answers with
source citations and retrieved context.

Usage:
    python cli.py              # Start interactive Q&A session
    python cli.py --top-k 3    # Use top 3 chunks for context
"""

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.pipeline import RAGPipeline
from src.config import TOP_K


console = Console()


def display_header():
    """Display the application header."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🤖 RAG Document Q&A Bot[/bold cyan]\n\n"
            "[dim]Ask questions about your document collection.\n"
            "The bot will answer using only information from the indexed documents.\n\n"
            "Commands:\n"
            "  [cyan]quit[/cyan] or [cyan]exit[/cyan]   — Exit the bot\n"
            "  [cyan]stats[/cyan]          — Show index statistics\n"
            "  [cyan]sources[/cyan]        — List indexed documents\n"
            "  [cyan]clear[/cyan]          — Clear the screen[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def display_answer(answer):
    """Display a generated answer with formatting and sources."""
    # Main answer
    console.print()
    console.print(
        Panel(
            Markdown(answer.answer),
            title="[bold green]💡 Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Source citations table
    if answer.sources:
        console.print()
        table = Table(
            title="📚 Retrieved Sources",
            border_style="blue",
            show_lines=True,
        )
        table.add_column("#", style="bold cyan", width=3, justify="center")
        table.add_column("Document", style="yellow", min_width=25)
        table.add_column("Page/Section", style="green", min_width=15)
        table.add_column("Relevance", style="magenta", justify="center", width=10)
        table.add_column("Excerpt", style="dim", max_width=50)

        for src in answer.sources:
            page_info = ""
            if src.get("section"):
                page_info = src["section"]
            elif src.get("page"):
                page_info = f"Page {src['page']}"

            score = src.get("score", 0)
            score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            score_str = f"{score:.0%} {score_bar}"

            excerpt = src.get("excerpt", "")
            if len(excerpt) > 80:
                excerpt = excerpt[:80] + "..."

            table.add_row(
                str(src.get("index", "")),
                src.get("source", "unknown"),
                page_info,
                score_str,
                excerpt,
            )

        console.print(table)

    console.print()


def display_stats(pipeline):
    """Display index statistics."""
    stats = pipeline.get_stats()

    table = Table(title="📊 Index Statistics", border_style="cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Indexed chunks", str(stats["indexed_chunks"]))
    table.add_row("Source documents", str(len(stats["sources"])))
    table.add_row("Index status", "✅ Ready" if stats["is_indexed"] else "❌ Empty")

    console.print()
    console.print(table)

    if stats["sources"]:
        console.print()
        console.print("[bold]📁 Indexed Documents:[/bold]")
        for source in stats["sources"]:
            console.print(f"   • {source}")

    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Document Q&A Bot"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K})",
    )
    args = parser.parse_args()

    display_header()

    # Initialize pipeline
    try:
        with console.status("[cyan]Initializing RAG pipeline...[/cyan]"):
            pipeline = RAGPipeline()
    except SystemExit:
        return

    # Check if documents are indexed
    if not pipeline.vector_store.is_indexed():
        console.print(
            Panel(
                "[bold red]❌ No documents indexed![/bold red]\n\n"
                "Please run the indexing step first:\n"
                "  [cyan]python index.py[/cyan]",
                border_style="red",
            )
        )
        return

    stats = pipeline.get_stats()
    console.print(
        f"[dim]📚 Ready! {stats['indexed_chunks']} chunks from "
        f"{len(stats['sources'])} documents indexed.[/dim]\n"
    )

    # Interactive loop
    while True:
        try:
            console.print("[bold cyan]?[/bold cyan] ", end="")
            question = input("Ask a question: ").strip()

            if not question:
                continue

            # Handle commands
            if question.lower() in ("quit", "exit", "q"):
                console.print("\n[dim]👋 Goodbye![/dim]\n")
                break

            if question.lower() == "stats":
                display_stats(pipeline)
                continue

            if question.lower() == "sources":
                sources = pipeline.vector_store.get_sources()
                console.print()
                console.print("[bold]📁 Indexed Documents:[/bold]")
                for s in sources:
                    console.print(f"   • {s}")
                console.print()
                continue

            if question.lower() == "clear":
                console.clear()
                display_header()
                continue

            # Process question
            with console.status("[cyan]🔍 Searching documents and generating answer...[/cyan]"):
                answer = pipeline.query(question, top_k=args.top_k)

            display_answer(answer)

        except KeyboardInterrupt:
            console.print("\n\n[dim]👋 Goodbye![/dim]\n")
            break

        except Exception as e:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}\n")


if __name__ == "__main__":
    main()

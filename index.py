#!/usr/bin/env python3
"""
Indexing Script — Document Ingestion Pipeline

Loads documents from the /data folder, chunks them, generates
embeddings, and stores everything in ChromaDB. This is a one-time
step that must be run before querying.

Usage:
    python index.py              # Index documents (skip if already done)
    python index.py --force      # Force re-index all documents
"""

import argparse
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.pipeline import RAGPipeline


console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Index documents for the RAG Q&A Bot"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if documents are already indexed",
    )
    args = parser.parse_args()

    # Header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]📚 RAG Document Q&A Bot — Indexing Pipeline[/bold cyan]\n"
            "[dim]Loading, chunking, embedding, and storing documents[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    start_time = time.time()

    try:
        pipeline = RAGPipeline()
    except SystemExit:
        return

    # Progress tracking
    stages = {
        "skip": "⏭️",
        "clear": "🗑️",
        "load": "📄",
        "chunk": "✂️",
        "embed": "🔢",
        "store": "💾",
    }

    current_stage = {"name": "", "message": ""}

    def progress_callback(stage: str, message: str):
        current_stage["name"] = stage
        current_stage["message"] = message
        icon = stages.get(stage, "•")
        console.print(f"  {icon}  {message}")

    try:
        result = pipeline.index(force=args.force, progress_callback=progress_callback)
    except Exception as e:
        console.print(f"\n[bold red]❌ Indexing failed:[/bold red] {e}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # Summary
    console.print()

    if result["status"] == "skipped":
        console.print(
            Panel.fit(
                f"[bold yellow]⏭️  Index already exists[/bold yellow]\n\n"
                f"Chunks: [bold]{result['chunks']}[/bold]\n"
                f"Documents: [bold]{result['documents']}[/bold]\n\n"
                f"[dim]Use --force to re-index[/dim]",
                border_style="yellow",
            )
        )
    else:
        # Show detailed results
        table = Table(title="📊 Indexing Summary", border_style="green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Documents processed", str(result["documents"]))
        table.add_row("Document sections", str(result.get("document_sections", "N/A")))
        table.add_row("Total chunks created", str(result["chunks"]))
        table.add_row("Time elapsed", f"{elapsed:.1f}s")

        console.print(table)

        # Show indexed sources
        console.print()
        console.print("[bold]📁 Indexed Documents:[/bold]")
        for source in result["sources"]:
            console.print(f"   • {source}")

        console.print()
        console.print(
            Panel.fit(
                "[bold green]✅ Indexing complete![/bold green]\n\n"
                "You can now run the Q&A bot:\n"
                "  [cyan]python cli.py[/cyan]       — Command-line interface\n"
                "  [cyan]streamlit run app.py[/cyan] — Web interface",
                border_style="green",
            )
        )

    console.print()


if __name__ == "__main__":
    main()

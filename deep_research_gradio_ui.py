"""
Gradio UI for the deep research pipeline.

This wraps the functions in `deep_research_colab_skeleton.py` in a
clean, streaming interface that:
- Accepts a research topic
- Shows step-by-step progress updates in real time
- Displays the sources (URLs and titles) used for the report
- Renders the final report in a readable markdown panel

To use in Colab or locally:
- Install gradio: `pip install gradio`
- Ensure `deep_research_colab_skeleton.py` is importable
- Run this file (or copy its contents into a Colab cell) and call `launch()`
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any, List, Tuple

import gradio as gr

from deep_research_colab_skeleton import (
    ModelPaths,
    ResearchConfig,
    FetchedPage,
    TextChunk,
    load_embedding_model,
    load_reranker_model,
    load_reasoning_llm,
    load_tiny_sd_pipeline,
    search_and_fetch,
    chunk_documents,
    embed_chunks,
    build_vector_index,
    retrieve,
    rerank,
    decompose_query_into_subquestions,
    generate_report,
)


def _chunks_to_source_rows(chunks: List[TextChunk]) -> List[Dict[str, Any]]:
    """
    Convert chunks into a de-duplicated list of sources for display.
    """
    seen = {}
    for c in chunks:
        key = c.url
        if not key:
            continue
        if key not in seen:
            seen[key] = {
                "url": c.url,
                "title": c.title,
            }
    return list(seen.values())


def research_workflow_stream(
    topic: str,
    root_path: str,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Streaming generator that runs the research pipeline and yields UI updates.

    Yields tuples of:
      - status markdown (what is happening now)
      - list of source dicts: {url, title}
      - report markdown (partial / final)

    This function assumes you have implemented the heavy lifting functions
    in `deep_research_colab_skeleton.py` (model loading, search, embeddings, etc.).
    """
    status_lines: List[str] = []
    sources: List[Dict[str, Any]] = []
    report_md = ""

    def push_status(line: str):
        status_lines.append(f"- {line}")

    push_status("Initializing models and configuration...")
    yield "\n".join(status_lines), sources, report_md

    model_paths = ModelPaths(root=root_path or ModelPaths().root)
    cfg = ResearchConfig()

    # 1) Load models
    push_status("Loading embedding model...")
    yield "\n".join(status_lines), sources, report_md
    embedding_model = load_embedding_model(model_paths)

    push_status("Loading reranker model...")
    yield "\n".join(status_lines), sources, report_md
    reranker_model = load_reranker_model(model_paths)

    push_status("Loading reasoning LLM (this may take a while)...")
    yield "\n".join(status_lines), sources, report_md
    reasoning_model = load_reasoning_llm(model_paths)

    # 2) Decompose query
    push_status("Decomposing topic into sub-questions...")
    yield "\n".join(status_lines), sources, report_md
    sub_questions = decompose_query_into_subquestions(topic, reasoning_model)

    push_status(f"Identified {len(sub_questions)} sub-questions.")
    yield "\n".join(status_lines), sources, report_md

    # 3) Search, fetch, and chunk for each sub-question
    all_pages: List[FetchedPage] = []
    for idx, sq in enumerate(sub_questions, start=1):
        push_status(f"[{idx}/{len(sub_questions)}] Searching web for: {sq!r}")
        yield "\n".join(status_lines), sources, report_md
        pages = search_and_fetch(sq, cfg)
        all_pages.extend(pages)

    push_status(f"Fetched {len(all_pages)} pages total. Chunking documents...")
    yield "\n".join(status_lines), sources, report_md
    chunks = chunk_documents(all_pages, cfg)

    # 4) Embed and build index
    push_status("Computing embeddings for chunks...")
    yield "\n".join(status_lines), sources, report_md
    embeddings = embed_chunks(chunks, embedding_model)

    push_status("Building vector index...")
    yield "\n".join(status_lines), sources, report_md
    index = build_vector_index(embeddings)

    # 5) Retrieve + rerank per sub-question
    push_status("Retrieving and reranking passages for each sub-question...")
    yield "\n".join(status_lines), sources, report_md
    selected_chunks: List[TextChunk] = []
    for idx, sq in enumerate(sub_questions, start=1):
        push_status(f"[{idx}/{len(sub_questions)}] Retrieving candidates for: {sq!r}")
        yield "\n".join(status_lines), sources, report_md
        candidates = retrieve(sq, chunks, embedding_model, index, cfg)

        push_status(f"[{idx}/{len(sub_questions)}] Reranking top candidates...")
        yield "\n".join(status_lines), sources, report_md
        reranked = rerank(sq, candidates, reranker_model, cfg)
        selected_chunks.extend(reranked)

        # Update visible sources in real time
        sources = _chunks_to_source_rows(selected_chunks)
        yield "\n".join(status_lines), sources, report_md

    # 6) Generate final report
    push_status("Generating structured report with citations...")
    yield "\n".join(status_lines), sources, report_md
    report_md = generate_report(topic, sub_questions, selected_chunks, reasoning_model, cfg)

    push_status("Done.")
    yield "\n".join(status_lines), sources, report_md


def launch():
    """
    Launch the Gradio UI.
    """
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Deep Research Assistant

            Enter a research topic, and the system will:

            - Break it into sub-questions
            - Search the web and collect sources
            - Retrieve and rerank the most relevant passages
            - Generate a structured, citation-rich report

            You can watch progress and sources update in real time.
            """
        )

        with gr.Row():
            topic_input = gr.Textbox(
                label="Research topic",
                placeholder="e.g., Long-term impacts of LLMs on software engineering (as of 2026)",
                lines=2,
            )
        with gr.Row():
            root_input = gr.Textbox(
                label="Model root path",
                value="/content/imagination-v1.1.0",
                info="Path where the `imagination-v1.1.0` folder is mounted (Colab or local).",
            )

        with gr.Row():
            run_button = gr.Button("Run deep research", variant="primary")

        with gr.Row():
            status_md = gr.Markdown(label="Progress")

        with gr.Row():
            sources_df = gr.Dataframe(
                headers=["url", "title"],
                datatype=["str", "str"],
                label="Sources used",
                interactive=False,
            )

        with gr.Row():
            report_md = gr.Markdown(label="Research report")

        run_button.click(
            fn=research_workflow_stream,
            inputs=[topic_input, root_input],
            outputs=[status_md, sources_df, report_md],
            api_name="research",
        )

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    launch()


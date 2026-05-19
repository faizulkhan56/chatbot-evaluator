from typing import Any, Dict

import streamlit as st

import api_client
from components import render_error, render_json, render_metric_row, render_result_path


st.set_page_config(page_title="Chatbot & RAG Evaluator", layout="wide")

PAGES = [
    "Home",
    "Chatbot Evaluation",
    "RAG Documents",
    "RAG Query",
    "RAG Evaluation",
    "Model Comparison",
    "Results Browser",
]


def main() -> None:
    st.sidebar.title("Evaluator")
    page = st.sidebar.radio("Page", PAGES)
    st.sidebar.caption(f"Backend: {api_client.FASTAPI_BASE_URL}")
    _render_sidebar_health()

    if page == "Home":
        render_home()
    elif page == "Chatbot Evaluation":
        render_chatbot_evaluation()
    elif page == "RAG Documents":
        render_rag_documents()
    elif page == "RAG Query":
        render_rag_query()
    elif page == "RAG Evaluation":
        render_rag_evaluation()
    elif page == "Model Comparison":
        render_model_comparison()
    elif page == "Results Browser":
        render_results_browser()


def _render_sidebar_health() -> None:
    try:
        health = api_client.get_health()
        st.sidebar.success(f"Backend {health.get('status', 'ok')}")
    except Exception:
        st.sidebar.error("Backend offline")


def render_home() -> None:
    st.title("Chatbot & RAG Evaluator")
    st.caption("Browser UI for local chatbot evaluation, RAG workflows, model comparison, and saved result browsing.")

    col1, col2, col3 = st.columns(3)
    try:
        health = api_client.get_health()
        datasets = api_client.list_datasets()
        results = api_client.list_results()
        col1.metric("Backend", health.get("status", "unknown"))
        col2.metric("Datasets", datasets.get("count", 0))
        col3.metric("Result files", len(results.get("results", [])))
    except Exception as exc:
        render_error(exc)

    st.info("Start FastAPI before using the frontend. Streamlit calls the backend API only and never calls OpenAI directly.")

    st.subheader("Run commands")
    st.code("uvicorn app:app --reload", language="bash")
    st.code("streamlit run frontend/streamlit_app.py", language="bash")

    st.subheader("Quick workflow")
    st.markdown(
        """
        1. Start the FastAPI backend.
        2. Upload or use documents under RAG Documents.
        3. Run RAG ingestion before RAG query or RAG evaluation.
        4. Run chatbot/RAG evaluations or model comparison.
        5. Browse saved JSON/CSV results.
        """
    )


def render_chatbot_evaluation() -> None:
    st.title("Chatbot Evaluation")
    st.caption("Build a small local dataset, run a selected chatbot model, and judge its answer against your reference answer.")

    if "chatbot_examples" not in st.session_state:
        st.session_state.chatbot_examples = []
    if "chatbot_eval_response" not in st.session_state:
        st.session_state.chatbot_eval_response = None

    st.subheader("Dataset Builder")
    dataset_name = st.text_input("Dataset name", value="ui_chatbot_eval", key="chatbot_builder_dataset_name")
    question = st.text_area("Question", value="", key="chatbot_builder_question")
    reference_answer = st.text_area("Reference answer", value="", key="chatbot_builder_reference")

    col_add, col_clear, col_save = st.columns(3)

    with col_add:
        if st.button("Add example", use_container_width=True):
            if not question.strip():
                st.warning("Question is required.")
            elif not reference_answer.strip():
                st.warning("Reference answer is required.")
            else:
                st.session_state.chatbot_examples.append(
                    {
                        "inputs": {"question": question.strip()},
                        "outputs": {"answer": reference_answer.strip()},
                    }
                )
                st.success("Example added.")

    with col_clear:
        if st.button("Clear examples", use_container_width=True):
            st.session_state.chatbot_examples = []
            st.session_state.chatbot_eval_response = None
            st.info("Examples cleared.")

    with col_save:
        if st.button("Save dataset", use_container_width=True):
            if not dataset_name.strip():
                st.warning("Dataset name is required.")
            elif not st.session_state.chatbot_examples:
                st.warning("Add at least one example before saving.")
            else:
                try:
                    response = api_client.create_dataset(dataset_name.strip(), st.session_state.chatbot_examples)
                    st.success(f"Dataset saved: {response.get('name', dataset_name)}")
                    render_json(response)
                except Exception as exc:
                    render_error(exc)

    if st.session_state.chatbot_examples:
        st.caption(f"Current examples: {len(st.session_state.chatbot_examples)}")
        st.dataframe(_examples_to_rows(st.session_state.chatbot_examples), use_container_width=True)
    else:
        st.info("No examples added yet.")

    st.divider()
    st.subheader("Evaluation Settings")
    eval_dataset_name = st.text_input("Evaluation dataset name", value=dataset_name or "ui_chatbot_eval")
    model_choice = st.selectbox("Model preset", ["gpt-4o-mini", "gpt-4.1-mini", "Custom"], index=0)
    model_name = (
        st.text_input("Custom model name", value="gpt-4o-mini")
        if model_choice == "Custom"
        else model_choice
    )
    evaluator_choice = st.selectbox("Evaluator model preset", ["gpt-4o-mini", "gpt-4.1-mini", "Custom"], index=0)
    evaluator_model = (
        st.text_input("Custom evaluator model name", value="gpt-4o-mini")
        if evaluator_choice == "Custom"
        else evaluator_choice
    )
    instructions = st.text_area(
        "Instructions",
        value="Respond to the user's question in a short, concise manner.",
    )
    concision_threshold = st.number_input("Concision threshold", min_value=1, value=30)
    save_result = st.checkbox("Save result", value=True)

    st.subheader("Run Evaluation")
    if st.button("Run Chatbot Evaluation", type="primary"):
        if not eval_dataset_name.strip():
            st.warning("Evaluation dataset name is required.")
            return
        payload = {
            "dataset_name": eval_dataset_name.strip(),
            "model_name": model_name,
            "evaluator_model": evaluator_model,
            "instructions": instructions,
            "concision_threshold": int(concision_threshold),
            "save_result": save_result,
        }
        try:
            result = api_client.evaluate_chatbot(payload)
            st.session_state.chatbot_eval_response = result
            st.success("Chatbot evaluation completed.")
        except Exception as exc:
            render_error(exc)

    if st.session_state.chatbot_eval_response:
        _render_chatbot_evaluation_result(st.session_state.chatbot_eval_response)


def render_rag_documents() -> None:
    st.title("RAG Documents")
    st.caption("Upload source files, review backend document storage, and build the in-memory RAG vector store.")

    if "documents_list_response" not in st.session_state:
        st.session_state.documents_list_response = None
    if "rag_ingest_response" not in st.session_state:
        st.session_state.rag_ingest_response = None

    st.warning(
        "The RAG vector store is currently in-memory. If you restart FastAPI, run ingestion again before RAG query or RAG evaluation."
    )

    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "Upload RAG source documents",
        type=["pdf", "csv", "xlsx", "xls", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Upload documents"):
        if not uploaded_files:
            st.warning("Select at least one document before uploading.")
            return

        try:
            files = [(file.name, file.getvalue(), file.type or "application/octet-stream") for file in uploaded_files]
            response = api_client.upload_documents(files)
            uploaded = response.get("uploaded_files", [])
            st.success(f"Uploaded {len(uploaded)} file(s).")
            if uploaded:
                st.dataframe(uploaded, use_container_width=True)
            st.session_state.documents_list_response = api_client.list_documents()
        except Exception as exc:
            render_error(exc)

    st.divider()
    st.subheader("Uploaded document list")
    if st.button("Refresh document list"):
        try:
            st.session_state.documents_list_response = api_client.list_documents()
        except Exception as exc:
            render_error(exc)

    if st.session_state.documents_list_response is None:
        try:
            st.session_state.documents_list_response = api_client.list_documents()
        except Exception as exc:
            render_error(exc)

    documents = (st.session_state.documents_list_response or {}).get("documents", [])
    if documents:
        st.dataframe(
            [
                {
                    "file_name": document.get("file_name"),
                    "path": document.get("path"),
                    "file_type": document.get("file_type"),
                    "size_bytes": document.get("size_bytes"),
                }
                for document in documents
            ],
            use_container_width=True,
        )
    else:
        st.info("No uploaded documents found.")

    st.divider()
    st.subheader("RAG ingestion settings")
    source_dir = st.text_input("Source directory", value="data/documents")
    chunk_size = st.number_input("Chunk size", min_value=1, value=500)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, value=50)
    embedding_model = st.text_input("Embedding model", value="text-embedding-3-small")

    if st.button("Ingest documents", type="primary"):
        if not source_dir.strip():
            st.warning("Source directory is required.")
            return
        if int(chunk_overlap) >= int(chunk_size):
            st.warning("Chunk overlap must be smaller than chunk size.")
            return

        payload = {
            "source_dir": source_dir.strip(),
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "embedding_model": embedding_model.strip(),
        }
        try:
            response = api_client.ingest_rag(payload)
            st.session_state.rag_ingest_response = response
            st.success("RAG ingestion completed.")
        except Exception as exc:
            render_error(exc)

    if st.session_state.rag_ingest_response:
        response = st.session_state.rag_ingest_response
        render_metric_row(
            {
                "Documents loaded": response.get("documents_loaded", 0),
                "Chunks created": response.get("chunks_created", 0),
                "Status": response.get("status", "unknown"),
            }
        )
        st.caption(f"Embedding model: {response.get('embedding_model', embedding_model)}")
        with st.expander("Raw ingest response"):
            render_json(response)


def render_rag_query() -> None:
    st.title("RAG Query")
    st.caption("Ask questions against the currently ingested documents and inspect the retrieved context.")

    if "rag_query_response" not in st.session_state:
        st.session_state.rag_query_response = None

    st.info(
        "Before running RAG query, upload and ingest documents from the RAG Documents page. "
        "The vector store is in-memory and must be rebuilt after backend restart."
    )

    question = st.text_area("Question", value="What is the main topic of the uploaded document?")
    model_choice = st.selectbox("Model preset", ["gpt-4o-mini", "gpt-4.1-mini", "Custom"], index=0, key="rag_query_model_choice")
    model_name = (
        st.text_input("Custom model name", value="gpt-4o-mini", key="rag_query_custom_model")
        if model_choice == "Custom"
        else model_choice
    )
    top_k = st.number_input("Top K", min_value=1, value=6)

    if st.button("Ask RAG", type="primary"):
        if not question.strip():
            st.warning("Question is required.")
            return
        payload = {"question": question, "model_name": model_name, "top_k": int(top_k)}
        try:
            result = api_client.query_rag(payload)
            st.session_state.rag_query_response = result
            st.success("RAG answer generated.")
        except Exception as exc:
            if "RAG vector store is not initialized" in str(exc):
                st.warning("Please go to RAG Documents page and run ingestion first.")
            render_error(exc)

    if st.session_state.rag_query_response:
        _render_rag_query_result(st.session_state.rag_query_response)


def render_rag_evaluation() -> None:
    st.title("RAG Evaluation")
    st.caption("Create a question/reference dataset and evaluate RAG answers for correctness, groundedness, and relevance.")

    if "rag_eval_examples" not in st.session_state:
        st.session_state.rag_eval_examples = []
    if "rag_eval_response" not in st.session_state:
        st.session_state.rag_eval_response = None

    st.info(
        "Before running RAG query or RAG evaluation, upload and ingest documents from the RAG Documents page. "
        "The vector store is in-memory and must be rebuilt after backend restart."
    )

    st.subheader("RAG Evaluation Dataset Builder")
    dataset_name = st.text_input("Dataset name", value="ui_rag_eval", key="rag_eval_builder_dataset_name")
    question = st.text_area(
        "Question",
        value="What is the main topic of the uploaded document?",
        key="rag_eval_builder_question",
    )
    reference_answer = st.text_area(
        "Reference answer",
        value="The document is about DevOps engineering experience and skills.",
        key="rag_eval_builder_reference",
    )

    col_add, col_clear, col_save = st.columns(3)
    with col_add:
        if st.button("Add RAG example", use_container_width=True):
            if not question.strip():
                st.warning("Question is required.")
            elif not reference_answer.strip():
                st.warning("Reference answer is required.")
            else:
                st.session_state.rag_eval_examples.append(
                    {
                        "inputs": {"question": question.strip()},
                        "outputs": {"answer": reference_answer.strip()},
                    }
                )
                st.success("RAG example added.")

    with col_clear:
        if st.button("Clear RAG examples", use_container_width=True):
            st.session_state.rag_eval_examples = []
            st.session_state.rag_eval_response = None
            st.info("RAG examples cleared.")

    with col_save:
        if st.button("Save RAG dataset", use_container_width=True):
            if not dataset_name.strip():
                st.warning("Dataset name is required.")
            elif not st.session_state.rag_eval_examples:
                st.warning("Add at least one RAG example before saving.")
            else:
                try:
                    response = api_client.create_dataset(dataset_name.strip(), st.session_state.rag_eval_examples)
                    st.success(f"RAG dataset saved: {response.get('name', dataset_name)}")
                    render_json(response)
                except Exception as exc:
                    render_error(exc)

    if st.session_state.rag_eval_examples:
        st.caption(f"Current RAG examples: {len(st.session_state.rag_eval_examples)}")
        st.dataframe(_examples_to_rows(st.session_state.rag_eval_examples), use_container_width=True)
    else:
        st.info("No RAG examples added yet.")

    st.divider()
    st.subheader("RAG Evaluation Settings")
    eval_dataset_name = st.text_input("Evaluation dataset name", value=dataset_name or "ui_rag_eval")
    model_choice = st.selectbox("Model preset", ["gpt-4o-mini", "gpt-4.1-mini", "Custom"], index=0, key="rag_eval_model_choice")
    model_name = (
        st.text_input("Custom model name", value="gpt-4o-mini", key="rag_eval_custom_model")
        if model_choice == "Custom"
        else model_choice
    )
    evaluator_choice = st.selectbox("Evaluator model preset", ["gpt-4o-mini", "gpt-4.1-mini", "Custom"], index=0, key="rag_eval_evaluator_choice")
    evaluator_model = (
        st.text_input("Custom evaluator model name", value="gpt-4o-mini", key="rag_eval_custom_evaluator")
        if evaluator_choice == "Custom"
        else evaluator_choice
    )
    top_k = st.number_input("Top K", min_value=1, value=6)
    save_result = st.checkbox("Save result", value=True)

    st.subheader("Run Evaluation")
    if st.button("Run RAG Evaluation", type="primary"):
        if not eval_dataset_name.strip():
            st.warning("Evaluation dataset name is required.")
            return
        payload = {
            "dataset_name": eval_dataset_name.strip(),
            "model_name": model_name,
            "evaluator_model": evaluator_model,
            "top_k": int(top_k),
            "save_result": save_result,
        }
        try:
            result = api_client.evaluate_rag(payload)
            st.session_state.rag_eval_response = result
            st.success("RAG evaluation completed.")
        except Exception as exc:
            if "RAG vector store is not initialized" in str(exc):
                st.warning("Please go to RAG Documents page and run ingestion first.")
            render_error(exc)

    if st.session_state.rag_eval_response:
        _render_rag_evaluation_result(st.session_state.rag_eval_response)


def render_model_comparison() -> None:
    st.title("Model Comparison")
    st.caption("Compare multiple models using the existing chatbot or RAG evaluation services.")

    if "compare_response" not in st.session_state:
        st.session_state.compare_response = None

    mode = st.selectbox("Mode", ["chatbot", "rag"])
    dataset_default = "chatbot_eval_sample" if mode == "chatbot" else "rag_eval_sample"
    dataset_name = st.text_input("Dataset name", value=dataset_default)
    models_text = st.text_input("Models, comma-separated", value="gpt-4o-mini,gpt-4.1-mini")
    evaluator_model = st.text_input("Evaluator model", value="gpt-4o-mini")
    instructions = None
    concision_threshold = 30
    top_k = 6

    if mode == "chatbot":
        instructions = st.text_area(
            "Instructions",
            value="Respond to the user's question in a short, concise manner.",
        )
        concision_threshold = st.number_input("Concision threshold", min_value=1, value=30)
    else:
        st.warning("For RAG comparison, upload and ingest documents first from the RAG Documents page.")
        top_k = st.number_input("Top K", min_value=1, value=6)

    save_result = st.checkbox("Save result", value=True)

    if st.button("Compare models", type="primary"):
        models = [model.strip() for model in models_text.split(",") if model.strip()]
        if not dataset_name.strip():
            st.warning("Dataset name is required.")
            return
        if not models:
            st.warning("Enter at least one model.")
            return

        payload = {
            "mode": mode,
            "dataset_name": dataset_name.strip(),
            "models": models,
            "evaluator_model": evaluator_model,
            "save_result": save_result,
        }
        if mode == "chatbot":
            payload["instructions"] = instructions
            payload["concision_threshold"] = int(concision_threshold)
        else:
            payload["top_k"] = int(top_k)

        try:
            result = api_client.compare_models(payload)
            st.session_state.compare_response = result
            st.success("Model comparison completed.")
        except Exception as exc:
            if "RAG vector store is not initialized" in str(exc):
                st.warning("Please go to RAG Documents page and run ingestion first.")
            render_error(exc)

    if st.session_state.compare_response:
        _render_compare_result(st.session_state.compare_response)


def render_results_browser() -> None:
    st.title("Results Browser")
    st.caption("List and inspect JSON/CSV evaluation artifacts saved by the backend.")

    if "results_list_response" not in st.session_state:
        st.session_state.results_list_response = None
    if "selected_result_response" not in st.session_state:
        st.session_state.selected_result_response = None

    if st.button("Refresh results"):
        try:
            st.session_state.results_list_response = api_client.list_results()
        except Exception as exc:
            render_error(exc)

    if st.session_state.results_list_response is None:
        try:
            st.session_state.results_list_response = api_client.list_results()
        except Exception as exc:
            render_error(exc)
            return

    results = st.session_state.results_list_response.get("results", [])
    if not results:
        st.info("No saved results found.")
        return

    st.subheader("Saved result files")
    st.dataframe(
        [
            {
                "file_name": result.get("file_name"),
                "path": result.get("path"),
                "file_type": result.get("file_type"),
                "size_bytes": result.get("size_bytes"),
            }
            for result in results
        ],
        use_container_width=True,
    )

    file_names = [result["file_name"] for result in results]
    selected = st.selectbox("Result file", file_names)
    if st.button("Load selected result", type="primary"):
        try:
            st.session_state.selected_result_response = api_client.get_result(selected)
            st.success(f"Loaded result file: {selected}")
        except Exception as exc:
            render_error(exc)

    if st.session_state.selected_result_response is not None:
        _render_loaded_result(selected, st.session_state.selected_result_response)


def _run_and_render(func, payload: Dict[str, Any]) -> None:
    try:
        result = func(payload)
        render_result_path(result)
        render_json(result)
    except Exception as exc:
        render_error(exc)


def _examples_to_rows(examples: list[dict]) -> list[dict]:
    return [
        {
            "question": example.get("inputs", {}).get("question", ""),
            "reference_answer": example.get("outputs", {}).get("answer", ""),
        }
        for example in examples
    ]


def _render_chatbot_evaluation_result(result: Dict[str, Any]) -> None:
    st.subheader("Metrics")
    summary = result.get("summary", {})
    render_metric_row(
        {
            "Total examples": result.get("total_examples", 0),
            "Correctness score": summary.get("correctness_score", 0),
            "Concision score": summary.get("concision_score", 0),
        }
    )

    st.subheader("Result table")
    rows = result.get("results", [])
    if rows:
        st.dataframe(
            [
                {
                    "question": row.get("question"),
                    "reference_answer": row.get("reference_answer"),
                    "model_response": row.get("model_response"),
                    "correctness": row.get("correctness"),
                    "concision": row.get("concision"),
                }
                for row in rows
            ],
            use_container_width=True,
        )

    render_result_path(result)

    with st.expander("Raw API Response"):
        render_json(result)


def _render_rag_query_result(result: Dict[str, Any]) -> None:
    st.subheader("Answer")
    answer = result.get("answer", "")
    if answer:
        st.success("RAG answer")
        st.markdown(answer)

    retrieved_documents = result.get("retrieved_documents", [])
    st.subheader("Retrieved documents")
    if not retrieved_documents:
        st.info("No retrieved documents returned.")
    for index, document in enumerate(retrieved_documents, start=1):
        metadata = document.get("metadata", {}) or {}
        source = metadata.get("source", "unknown")
        page = metadata.get("page")
        title = f"Document {index}: {source}"
        if page is not None:
            title += f" | page {page}"
        with st.expander(title):
            st.caption(f"Source: {source}")
            if page is not None:
                st.caption(f"Page: {page}")
            content = document.get("content", "")
            st.text_area(
                "Content preview",
                value=content[:2000],
                height=220,
                key=f"rag_query_doc_{index}",
                disabled=True,
            )

    with st.expander("Raw API Response"):
        render_json(result)


def _render_rag_evaluation_result(result: Dict[str, Any]) -> None:
    st.subheader("Metrics")
    summary = result.get("summary", {})
    render_metric_row(
        {
            "Total examples": result.get("total_examples", 0),
            "Correctness score": summary.get("correctness_score", 0),
            "Groundedness score": summary.get("groundedness_score", 0),
            "Answer relevance score": summary.get("relevance_score", 0),
            "Retrieval relevance score": summary.get("retrieval_relevance_score", 0),
        }
    )

    st.subheader("Result table")
    rows = result.get("results", [])
    if rows:
        st.dataframe(
            [
                {
                    "question": row.get("question"),
                    "reference_answer": row.get("reference_answer"),
                    "rag_answer": row.get("rag_answer"),
                    "correctness": row.get("correctness"),
                    "groundedness": row.get("groundedness"),
                    "relevance": row.get("relevance"),
                    "retrieval_relevance": row.get("retrieval_relevance"),
                    "retrieved_docs_count": row.get("retrieved_docs_count"),
                }
                for row in rows
            ],
            use_container_width=True,
        )

    render_result_path(result)

    with st.expander("Raw API Response"):
        render_json(result)


def _render_compare_result(result: Dict[str, Any]) -> None:
    st.subheader("Summary by model")
    summary_by_model = result.get("summary_by_model", [])
    if summary_by_model:
        st.dataframe(summary_by_model, use_container_width=True)

        best = _best_model_by_correctness(summary_by_model)
        if best:
            render_metric_row(
                {
                    "Models compared": len(summary_by_model),
                    "Best correctness": best.get("correctness_score", 0),
                    "Best model": best.get("model_name", "n/a"),
                }
            )

    st.subheader("Detailed results by model")
    for model_group in result.get("results", []):
        model_name = model_group.get("model_name", "unknown")
        rows = model_group.get("results", [])
        with st.expander(f"{model_name} ({len(rows)} row(s))"):
            if rows:
                st.dataframe(rows, use_container_width=True)
            else:
                st.info("No detailed rows returned.")

    render_result_path(result)

    with st.expander("Raw API Response"):
        render_json(result)


def _render_loaded_result(file_name: str, result: Any) -> None:
    st.subheader("Loaded result")
    if isinstance(result, list):
        st.caption(f"CSV rows from {file_name}")
        if result:
            st.dataframe(result, use_container_width=True)
        else:
            st.info("CSV file has no rows.")
        with st.expander("Raw response"):
            render_json(result)
        return

    if isinstance(result, dict):
        st.caption(f"JSON result from {file_name}")
        summary = result.get("summary")
        summary_by_model = result.get("summary_by_model")
        if isinstance(summary, dict):
            st.subheader("Summary")
            st.dataframe([summary], use_container_width=True)
        if isinstance(summary_by_model, list) and summary_by_model:
            st.subheader("Summary by model")
            st.dataframe(summary_by_model, use_container_width=True)

        rows = result.get("results")
        if isinstance(rows, list) and rows:
            st.subheader("Results")
            if all(isinstance(item, dict) and "model_name" in item and "results" in item for item in rows):
                for group in rows:
                    with st.expander(f"{group.get('model_name', 'unknown')} ({len(group.get('results', []))} row(s))"):
                        if group.get("results"):
                            st.dataframe(group["results"], use_container_width=True)
                        else:
                            st.info("No detailed rows returned.")
            else:
                st.dataframe(rows, use_container_width=True)

        with st.expander("Raw response"):
            render_json(result)
        return

    st.write(result)


def _best_model_by_correctness(summary_by_model: list[dict]) -> dict | None:
    candidates = [row for row in summary_by_model if row.get("correctness_score") is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row.get("correctness_score", 0))


if __name__ == "__main__":
    main()

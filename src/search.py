import os
from dotenv import load_dotenv

from src.vectorstore import FaissVectorStore
from langchain_openai import ChatOpenAI

# Groq support kept for future use
# from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load existing vector store if available, otherwise build it
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents

            docs = load_all_documents("data")

            if not docs:
                raise ValueError("No documents found in data folder. Please add files inside data/.")

            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # -----------------------------
        # OpenAI LLM configuration
        # -----------------------------
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing. Please add it to your .env file.")

        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=llm_model,
        )

        print(f"[INFO] OpenAI LLM initialized: {llm_model}")

        # -----------------------------
        # Groq LLM configuration
        # Uncomment this block if you want to use Groq later
        # -----------------------------
        # groq_api_key = os.getenv("GROQ_API_KEY")
        #
        # if not groq_api_key:
        #     raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")
        #
        # self.llm = ChatGroq(
        #     groq_api_key=groq_api_key,
        #     model_name="gemma2-9b-it",
        # )
        #
        # print("[INFO] Groq LLM initialized: gemma2-9b-it")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)

        texts = [
            result["metadata"].get("text", "")
            for result in results
            if result.get("metadata")
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""
Summarize the following context for the query: '{query}'

Context:
{context}

Summary:
"""

        response = self.llm.invoke(prompt)
        return response.content


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
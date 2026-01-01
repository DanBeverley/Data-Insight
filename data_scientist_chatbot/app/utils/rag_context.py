import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RAGContextManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.client = chromadb.Client()
        self.collection_name = f"session_{session_id}"

        # Use a lightweight local embedding model
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, embedding_function=self.ef
            )
        except Exception as e:
            logger.error(f"Failed to create ChromaDB collection: {e}")
            self.collection = None

    def index_dataset(self, df: pd.DataFrame, profile: Dict[str, Any] = None):
        """Index dataset columns and profile info into Vector DB"""
        if not self.collection:
            return

        documents = []
        metadatas = []
        ids = []

        # 1. Index Columns
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_values = str(df[col].dropna().sample(min(5, len(df))).tolist())

            # Create a semantic description of the column
            doc_text = f"Column: {col}, Type: {dtype}, Samples: {sample_values}"

            # Handle profile whether it's a dict or DataProfileSummary object
            ai_context = {}
            if profile:
                if isinstance(profile, dict):
                    ai_context = profile.get("ai_agent_context", {})
                elif hasattr(profile, "ai_agent_context"):
                    ai_context = profile.ai_agent_context

            if ai_context and "column_details" in ai_context:
                details = ai_context["column_details"].get(col, {})
                semantic_type = details.get("semantic_type", "")
                doc_text += f", Semantic Type: {semantic_type}"

            documents.append(doc_text)
            metadatas.append({"type": "column", "name": col, "dtype": dtype})
            ids.append(f"col_{col}")

        # 2. Index Profile Insights (if available)
        if profile:
            # Add general summary
            documents.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            metadatas.append({"type": "summary", "name": "shape"})
            ids.append("summary_shape")

        # Add to ChromaDB
        try:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Indexed {len(documents)} items for RAG context")
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")

    def index_learning_patterns(self, patterns: List[Dict[str, Any]]):
        """Index successful learning patterns (code, insights) into Vector DB"""
        if not self.collection or not patterns:
            return

        documents = []
        metadatas = []
        ids = []

        for p in patterns:
            # Pattern structure from SQLite: {'type': ..., 'data': {...}, 'success_count': ..., 'confidence': ...}
            p_type = p.get("type", "unknown")
            p_data = p.get("data", {})

            # Construct semantic document
            if p_type == "code_execution":
                request = p_data.get("request", "")
                code = p_data.get("code_snippet", "")
                doc_text = f"Task: {request}\nSolution Code:\n{code}"
            else:
                doc_text = str(p_data)

            documents.append(doc_text)
            metadatas.append(
                {
                    "type": "pattern",
                    "pattern_type": p_type,
                    "confidence": p.get("confidence", 0.0),
                    "success_count": p.get("success_count", 0),
                }
            )
            # Use hash or unique ID
            import hashlib

            p_id = hashlib.md5(doc_text.encode()).hexdigest()
            ids.append(f"pat_{p_id}")

        try:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Indexed {len(documents)} learning patterns")
        except Exception as e:
            logger.error(f"Failed to index patterns: {e}")

    def retrieve_similar_patterns(self, query: str, n_results: int = 3) -> List[Dict]:
        """Retrieve similar successful patterns for a given query"""
        if not self.collection:
            return []

        try:
            results = self.collection.query(
                query_texts=[query], n_results=n_results, where={"type": "pattern"}  # Filter for patterns only
            )

            patterns = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i]
                    patterns.append(
                        {"content": doc, "type": meta.get("pattern_type"), "confidence": meta.get("confidence")}
                    )
            return patterns

        except Exception as e:
            logger.error(f"Pattern retrieval failed: {e}")
            return []

    def retrieve_context(self, query: str, n_results: int = 10) -> str:
        """Retrieve relevant context (columns + patterns) based on user query"""
        if not self.collection:
            return "RAG System unavailable."

        try:
            # 1. Retrieve Data Context (Columns/Profile)
            data_results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"type": {"$in": ["column", "summary"]}},  # Filter for data context
            )

            context_lines = ["RELEVANT DATA CONTEXT (RAG-Retrieval):"]
            if data_results["documents"]:
                for i, doc in enumerate(data_results["documents"][0]):
                    context_lines.append(f"• {doc}")

            # 2. Retrieve Similar Patterns (Historical Learning)
            patterns = self.retrieve_similar_patterns(query, n_results=3)
            if patterns:
                context_lines.append("\nRELEVANT PAST SOLUTIONS (Memory):")
                for p in patterns:
                    context_lines.append(f"• [Confidence: {p['confidence']:.2f}] {p['content'][:200]}...")

            return "\n".join(context_lines)

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return "Error retrieving context."

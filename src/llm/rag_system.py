"""
RAG (Retrieval-Augmented Generation) System for DataInsight AI
Provides context-aware information retrieval for enhanced conversational AI
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sqlite3
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document structure for RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    timestamp: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    document: Document
    score: float
    relevance: str  # 'high', 'medium', 'low'


class LocalRAGSystem:
    """
    Local RAG system using vector embeddings and SQLite storage
    Provides context retrieval without external dependencies
    """
    
    def __init__(self, db_path: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path or "data_insight_rag.db"
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.documents = {}
        self.embeddings = {}
        
        self._initialize_database()
        self._load_embedding_model()
        self._load_documents()
        
    @staticmethod
    def _json_default(o):
        """Convert numpy and datetime types to JSON-serializable values."""
        try:
            import numpy as _np  # local alias to avoid shadowing
            from datetime import datetime as _dt
            if isinstance(o, (_np.float32, _np.float64)):
                return float(o)
            if isinstance(o, (_np.int32, _np.int64)):
                return int(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
            if isinstance(o, _dt):
                return o.isoformat()
        except Exception:
            pass
        return str(o)

    def _initialize_database(self):
        """Initialize SQLite database for document storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB,
                    timestamp TEXT,
                    document_type TEXT,
                    session_id TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    results TEXT,
                    timestamp TEXT,
                    session_id TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"RAG database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            
    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Embedding model {self.embedding_model_name} loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            self.embedding_model = None
            
    def _load_documents(self):
        """Load existing documents from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, content, metadata, embedding, timestamp FROM documents")
            rows = cursor.fetchall()
            
            for row in rows:
                doc_id, content, metadata_str, embedding_blob, timestamp = row
                
                metadata = json.loads(metadata_str) if metadata_str else {}
                embedding = None
                if embedding_blob:
                    try:
                        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                        # Ensure consistent embedding dimensions (384 for all-MiniLM-L6-v2)
                        if len(embedding) != 384:
                            logger.warning(f"Embedding dimension mismatch for doc {doc_id}: {len(embedding)} != 384")
                            embedding = None  # Skip this embedding, will regenerate if needed
                    except Exception as e:
                        logger.warning(f"Failed to load embedding for doc {doc_id}: {e}")
                        embedding = None
                
                document = Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    embedding=embedding,
                    timestamp=timestamp
                )
                
                self.documents[doc_id] = document
                if embedding is not None:
                    self.embeddings[doc_id] = embedding
            
            conn.close()
            logger.info(f"Loaded {len(self.documents)} documents from database")
            
        except Exception as e:
            logger.error(f"Document loading failed: {e}")
            
    def add_document(self, content: str, metadata: Dict[str, Any] = None, 
                    document_type: str = "general", session_id: str = None) -> str:
        """Add a document to the RAG system"""
        try:
            # Generate document ID
            doc_id = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()
            
            # Generate embedding if model available
            embedding = None
            if self.embedding_model:
                embedding = self.embedding_model.encode([content])[0].astype(np.float32)
                self.embeddings[doc_id] = embedding
            
            # Create document
            document = Document(
                id=doc_id,
                content=content,
                metadata=metadata or {},
                embedding=embedding,
                timestamp=datetime.now().isoformat()
            )
            
            self.documents[doc_id] = document
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (id, content, metadata, embedding, timestamp, document_type, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_id,
                content,
                json.dumps(metadata or {}),
                embedding.tobytes() if embedding is not None else None,
                document.timestamp,
                document_type,
                session_id
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Document {doc_id} added successfully")
            return doc_id
            
        except Exception as e:
            logger.error(f"Document addition failed: {e}")
            return None
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5, 
                                  session_id: str = None) -> List[RetrievalResult]:
        """Retrieve most relevant documents for a query"""
        try:
            if not self.embedding_model or not self.embeddings:
                # Fallback to keyword-based search
                return self._keyword_search(query, top_k)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                if doc_id in self.documents:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        doc_embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append((doc_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Create retrieval results
            results = []
            for doc_id, score in similarities[:top_k]:
                document = self.documents[doc_id]
                
                # Determine relevance level
                if score > 0.7:
                    relevance = "high"
                elif score > 0.4:
                    relevance = "medium"
                else:
                    relevance = "low"
                
                results.append(RetrievalResult(
                    document=document,
                    score=score,
                    relevance=relevance
                ))
            
            # Log query
            self._log_query(query, results, session_id)
            
            return results
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def _keyword_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Fallback keyword-based search when embeddings unavailable"""
        query_words = set(query.lower().split())
        
        scored_documents = []
        for doc_id, document in self.documents.items():
            content_words = set(document.content.lower().split())
            
            # Simple intersection-based scoring
            intersection = query_words.intersection(content_words)
            score = len(intersection) / len(query_words) if query_words else 0
            
            if score > 0:
                scored_documents.append((doc_id, score))
        
        # Sort and limit
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in scored_documents[:top_k]:
            document = self.documents[doc_id]
            relevance = "high" if score > 0.5 else "medium" if score > 0.2 else "low"
            
            results.append(RetrievalResult(
                document=document,
                score=score,
                relevance=relevance
            ))
        
        return results
    
    def _log_query(self, query: str, results: List[RetrievalResult], session_id: str = None):
        """Log query and results for analytics"""
        try:
            query_id = hashlib.md5(f"{query}{datetime.now().isoformat()}".encode()).hexdigest()
            
            results_data = [
                {
                    "document_id": result.document.id,
                    # Ensure JSON-serializable primitive
                    "score": float(result.score),
                    "relevance": result.relevance
                }
                for result in results
            ]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO queries (id, query, results, timestamp, session_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                query_id,
                query,
                json.dumps(results_data, default=self._json_default),
                datetime.now().isoformat(),
                session_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Query logging failed: {e}")
    
    def add_execution_report(self, execution_result: Dict[str, Any], session_id: str = None):
        """Add pipeline execution report to RAG system"""
        try:
            # Extract key information from execution result
            summary = execution_result.get("execution_summary", {})
            
            report_content = f"""
Pipeline Execution Report:
- Status: {execution_result.get('status', 'unknown')}
- Execution Time: {summary.get('total_time', 'unknown')} seconds
- Successful Stages: {summary.get('successful_stages', 0)} / {summary.get('total_stages', 0)}
- Task Type: {execution_result.get('task_type', 'unknown')}

Results Summary:
- Data processed successfully: {execution_result.get('status') == 'success'}
- Pipeline type: {execution_result.get('pipeline_type', 'unknown')}
"""
            
            if 'results' in execution_result:
                results = execution_result['results']
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, dict) and 'shape' in value:
                            report_content += f"- {key} shape: {value['shape']}\n"
            
            metadata = {
                "type": "execution_report",
                "status": execution_result.get("status"),
                "execution_id": execution_result.get("execution_id"),
                "task_type": execution_result.get("task_type"),
                "timestamp": datetime.now().isoformat()
            }
            
            doc_id = self.add_document(
                content=report_content,
                metadata=metadata,
                document_type="execution_report",
                session_id=session_id
            )
            
            logger.info(f"Execution report added to RAG system: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Execution report addition failed: {e}")
            return None
    
    def add_intelligence_profile(self, profile: Dict[str, Any], session_id: str = None):
        """Add intelligence profile to RAG system"""
        try:
            # Extract key insights from profile
            profile_content = "Data Intelligence Profile:\n"
            
            if 'column_profiles' in profile:
                profile_content += f"- Analyzed {len(profile['column_profiles'])} columns\n"
                
                semantic_types = {}
                for col, col_profile in profile['column_profiles'].items():
                    if isinstance(col_profile, dict):
                        semantic_type = col_profile.get('semantic_type', 'unknown')
                        semantic_types[semantic_type] = semantic_types.get(semantic_type, 0) + 1
                
                profile_content += "- Semantic types detected:\n"
                for sem_type, count in semantic_types.items():
                    profile_content += f"  * {sem_type}: {count} columns\n"
            
            if 'domain_analysis' in profile:
                domain_analysis = profile['domain_analysis']
                profile_content += f"- Domain detected: {domain_analysis.get('detected_domain', 'unknown')}\n"
                profile_content += f"- Domain confidence: {domain_analysis.get('confidence', 0):.2f}\n"
            
            if 'relationship_analysis' in profile:
                relationships = profile['relationship_analysis'].get('relationships', [])
                profile_content += f"- Found {len(relationships)} data relationships\n"
            
            if 'overall_recommendations' in profile:
                recommendations = profile['overall_recommendations']
                profile_content += f"- Generated {len(recommendations)} recommendations\n"
            
            metadata = {
                "type": "intelligence_profile",
                "columns_analyzed": len(profile.get('column_profiles', {})),
                "domain": profile.get('domain_analysis', {}).get('detected_domain'),
                "timestamp": datetime.now().isoformat()
            }
            
            doc_id = self.add_document(
                content=profile_content,
                metadata=metadata,
                document_type="intelligence_profile",
                session_id=session_id
            )
            
            logger.info(f"Intelligence profile added to RAG system: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Intelligence profile addition failed: {e}")
            return None
    
    def get_context_for_query(self, query: str, session_id: str = None) -> str:
        """Get contextual information for a query"""
        try:
            # Retrieve relevant documents
            results = self.retrieve_relevant_documents(query, top_k=3, session_id=session_id)
            
            if not results:
                return "No relevant context found."
            
            # Build context string
            context_parts = ["Relevant context:"]
            
            for i, result in enumerate(results[:3], 1):
                context_parts.append(f"\n{i}. {result.document.content[:300]}...")
                if result.document.metadata.get('type'):
                    context_parts.append(f"   (Type: {result.document.metadata['type']})")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Context generation failed: {e}")
            return "Context generation failed."
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get all context for a specific session"""
        try:
            session_docs = {
                doc_id: doc for doc_id, doc in self.documents.items()
                if doc.metadata.get('session_id') == session_id or session_id in str(doc.metadata)
            }
            
            # Categorize documents by type
            context = {
                "execution_reports": [],
                "intelligence_profiles": [],
                "general_documents": [],
                "total_documents": len(session_docs)
            }
            
            for doc in session_docs.values():
                doc_type = doc.metadata.get('type', 'general')
                if doc_type == 'execution_report':
                    context["execution_reports"].append({
                        "content": doc.content[:200] + "...",
                        "metadata": doc.metadata,
                        "timestamp": doc.timestamp
                    })
                elif doc_type == 'intelligence_profile':
                    context["intelligence_profiles"].append({
                        "content": doc.content[:200] + "...",
                        "metadata": doc.metadata,
                        "timestamp": doc.timestamp
                    })
                else:
                    context["general_documents"].append({
                        "content": doc.content[:200] + "...",
                        "metadata": doc.metadata,
                        "timestamp": doc.timestamp
                    })
            
            return context
            
        except Exception as e:
            logger.error(f"Session context retrieval failed: {e}")
            return {"error": str(e)}
    
    def cleanup_old_documents(self, days: int = 30):
        """Clean up documents older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM documents WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            
            cursor.execute(
                "DELETE FROM queries WHERE timestamp < ?", 
                (cutoff_date.isoformat(),)
            )
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            # Update in-memory storage
            to_remove = [
                doc_id for doc_id, doc in self.documents.items()
                if doc.timestamp and datetime.fromisoformat(doc.timestamp) < cutoff_date
            ]
            
            for doc_id in to_remove:
                del self.documents[doc_id]
                if doc_id in self.embeddings:
                    del self.embeddings[doc_id]
            
            logger.info(f"Cleaned up {deleted_count} old documents")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Document cleanup failed: {e}")
            return 0
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get RAG system analytics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Document statistics
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT document_type, COUNT(*) FROM documents GROUP BY document_type")
            doc_types = dict(cursor.fetchall())
            
            # Query statistics
            cursor.execute("SELECT COUNT(*) FROM queries")
            total_queries = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count 
                FROM queries 
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            """)
            recent_queries = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_documents": total_docs,
                "document_types": doc_types,
                "total_queries": total_queries,
                "queries_last_7_days": recent_queries,
                "embeddings_available": len(self.embeddings),
                "database_path": self.db_path,
                "embedding_model": self.embedding_model_name
            }
            
        except Exception as e:
            logger.error(f"Analytics generation failed: {e}")
            return {"error": str(e)}
"""
Vector database and retrieval system for soccer data RAG pipeline.
"""

import logging
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class SoccerRAGRetriever:
    """Handles vector search and context retrieval for soccer queries."""
    
    def __init__(self, collection_name: str = "statsbomb_soccer", db_path: str = "./chroma_db"):
        """
        Initialize the retriever with ChromaDB and sentence transformer.
        
        Args:
            collection_name: Name of the ChromaDB collection
            db_path: Path to store ChromaDB data
        """
        self.collection_name = collection_name
        self.db_path = db_path
        
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            logger.info(f"Connected to ChromaDB at {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
        
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Using collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to create/get collection: {e}")
            raise
        
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add soccer data documents to vector database.
        
        Args:
            documents: List of document dictionaries with id, content, metadata
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        try:
            ids = [doc["id"] for doc in documents]
            contents = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            logger.info(f"Encoding {len(contents)} documents...")
            embeddings = self.encoder.encode(contents, show_progress_bar=True)
            
            logger.info(f"Adding {len(documents)} documents to ChromaDB...")
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=contents,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def semantic_search(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Perform semantic search across soccer data.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            
        Returns:
            List of search results with content and metadata
        """
        try:
            query_embedding = self.encoder.encode([query])
            
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, filters: Optional[Dict] = None, n_results: int = 10) -> List[Dict]:
        """
        Combine semantic search with metadata filtering.
        
        Args:
            query: Search query string
            filters: Dictionary of metadata filters
            n_results: Number of results to return
            
        Returns:
            List of filtered search results
        """
        try:
            query_embedding = self.encoder.encode([query])
            
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if value is not None and value != "":
                        where_clause[key] = value
            
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
            
            filter_desc = f" with filters {filters}" if filters else ""
            logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...{filter_desc}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"total_documents": 0, "collection_name": self.collection_name}
    
    def search_by_competition(self, query: str, competition: str, n_results: int = 10) -> List[Dict]:
        """Search within a specific competition."""
        return self.hybrid_search(
            query=query,
            filters={"competition": competition},
            n_results=n_results
        )
    
    def search_by_team(self, query: str, team: str, n_results: int = 10) -> List[Dict]:
        """Search for content related to a specific team."""
        filters = {
            "$or": [
                {"home_team": team},
                {"away_team": team},
                {"team_name": team}
            ]
        }
        
        try:
            query_embedding = self.encoder.encode([query])
            
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results * 2
            )
            
            filtered_results = []
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                if (metadata.get('home_team') == team or 
                    metadata.get('away_team') == team or 
                    metadata.get('team_name') == team):
                    
                    result = {
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": metadata,
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    }
                    filtered_results.append(result)
                    
                    if len(filtered_results) >= n_results:
                        break
            
            logger.info(f"Found {len(filtered_results)} results for team {team}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Team search failed: {e}")
            return []
    
    def search_by_player(self, query: str, player: str, n_results: int = 10) -> List[Dict]:
        """Search for content related to a specific player."""
        return self.hybrid_search(
            query=query,
            filters={"player_name": player},
            n_results=n_results
        )
    
    def get_recent_matches(self, n_results: int = 20) -> List[Dict]:
        """Get recent match information."""
        try:
            results = self.collection.query(
                query_embeddings=None,
                n_results=n_results,
                where={"type": "match_metadata"}
            )
            
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i]
                }
                formatted_results.append(result)
            
            return sorted(formatted_results, 
                         key=lambda x: x['metadata'].get('match_date', ''), 
                         reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get recent matches: {e}")
            return []

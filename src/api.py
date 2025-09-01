"""
FastAPI backend for StatsBomb RAG chatbot system.
"""

import logging
import os
import uuid
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .rag_retrieval import SoccerRAGRetriever
from .rag_generation import SoccerRAGGenerator, ConversationMemory
from .data_loader import StatsBombDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="StatsBomb RAG Chatbot API",
    description="RAG-powered chatbot for StatsBomb soccer data analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict]
    session_id: str


class CompetitionInfo(BaseModel):
    competition_id: int
    competition_name: str
    season_name: str
    match_available: Optional[str] = None


retriever: Optional[SoccerRAGRetriever] = None
generator: Optional[SoccerRAGGenerator] = None
memory_store: Dict[str, ConversationMemory] = {}
data_loader: Optional[StatsBombDataLoader] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup."""
    global retriever, generator, data_loader
    
    try:
        logger.info("Initializing RAG components...")
        
        retriever = SoccerRAGRetriever()
        logger.info("✓ Vector retriever initialized")
        
        generator = SoccerRAGGenerator()
        logger.info("✓ Response generator initialized")
        
        data_loader = StatsBombDataLoader()
        logger.info("✓ Data loader initialized")
        
        stats = retriever.get_collection_stats()
        logger.info(f"✓ Vector database ready with {stats['total_documents']} documents")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        raise


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for soccer queries.
    
    Args:
        request: Chat request with query and optional session_id
        
    Returns:
        Chat response with generated answer and sources
    """
    if not retriever or not generator:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        if session_id not in memory_store:
            memory_store[session_id] = ConversationMemory()
        
        memory = memory_store[session_id]
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        context_docs = retriever.semantic_search(request.query, n_results=8)
        
        conversation_history = memory.get_relevant_history(request.query)
        
        response_text = generator.generate_response(
            query=request.query,
            context_docs=context_docs,
            conversation_history=conversation_history
        )
        
        memory.add_exchange(
            user_query=request.query,
            assistant_response=response_text,
            context_used=context_docs
        )
        
        sources = [
            {
                "content": doc.get("content", "")[:200] + "...",
                "metadata": doc.get("metadata", {}),
                "relevance_score": 1.0 - (doc.get("distance", 0) or 0)
            }
            for doc in context_docs[:5]
        ]
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat request")


@app.get("/competitions", response_model=List[CompetitionInfo])
async def get_competitions():
    """Get available competitions in the dataset."""
    if not data_loader:
        raise HTTPException(status_code=500, detail="Data loader not initialized")
    
    try:
        competitions = data_loader.get_all_competitions()
        
        competition_info = [
            CompetitionInfo(
                competition_id=comp["competition_id"],
                competition_name=comp["competition_name"],
                season_name=comp["season_name"],
                match_available=comp.get("match_available")
            )
            for comp in competitions
        ]
        
        return competition_info
        
    except Exception as e:
        logger.error(f"Failed to get competitions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve competitions")


@app.get("/search")
async def search_endpoint(
    query: str,
    competition: Optional[str] = None,
    team: Optional[str] = None,
    player: Optional[str] = None,
    limit: int = 10
):
    """
    Direct search endpoint for testing retrieval.
    
    Args:
        query: Search query
        competition: Filter by competition
        team: Filter by team
        player: Filter by player
        limit: Number of results to return
    """
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    try:
        if player:
            results = retriever.search_by_player(query, player, limit)
        elif team:
            results = retriever.search_by_team(query, team, limit)
        elif competition:
            results = retriever.search_by_competition(query, competition, limit)
        else:
            results = retriever.semantic_search(query, limit)
        
        return {
            "query": query,
            "filters": {
                "competition": competition,
                "team": team,
                "player": player
            },
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    if not retriever:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        stats = retriever.get_collection_stats()
        
        return {
            "vector_database": stats,
            "active_sessions": len(memory_store),
            "system_status": "healthy"
        }
        
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if session_id in memory_store:
        memory_store[session_id].clear_history()
        return {"message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "retriever": retriever is not None,
            "generator": generator is not None,
            "data_loader": data_loader is not None
        }
    }


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

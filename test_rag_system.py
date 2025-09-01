"""
Test script for the complete RAG system.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import StatsBombDataLoader
from rag_ingestion import SoccerDataChunker
from rag_retrieval import SoccerRAGRetriever
from rag_generation import SoccerRAGGenerator, ConversationMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rag_pipeline():
    """Test the complete RAG pipeline with sample data."""
    
    print("🚀 Testing StatsBomb RAG System")
    print("=" * 50)
    
    print("\n1. Testing Data Loading...")
    loader = StatsBombDataLoader()
    competitions = loader.get_all_competitions()
    print(f"✅ Found {len(competitions)} competitions")
    
    wsl_comp = next((c for c in competitions if c['competition_id'] == 37), None)
    if not wsl_comp:
        print("❌ FA Women's Super League not found")
        return
    
    print(f"✅ Using {wsl_comp['competition_name']} for testing")
    
    matches_df = loader.download_matches(37, 90)  # 2020/2021 season
    print(f"✅ Loaded {len(matches_df)} matches")
    
    print("\n2. Testing Data Chunking...")
    chunker = SoccerDataChunker()
    match_chunks = chunker.chunk_match_metadata(matches_df)
    print(f"✅ Created {len(match_chunks)} match chunks")
    
    if len(matches_df) > 0:
        sample_match = matches_df.iloc[0]
        match_id = sample_match['match_id']
        print(f"✅ Loading events for match {match_id}")
        
        try:
            events_df = loader.download_events(match_id)
            if not events_df.empty:
                match_info = sample_match.to_dict()
                event_chunks = chunker.chunk_match_events(events_df, match_info)
                print(f"✅ Created {len(event_chunks)} event chunks")
                match_chunks.extend(event_chunks)
            else:
                print("⚠️ No events found for sample match")
        except Exception as e:
            print(f"⚠️ Could not load events: {e}")
    
    print("\n3. Testing Vector Database...")
    retriever = SoccerRAGRetriever()
    
    if match_chunks:
        print(f"Adding {len(match_chunks)} chunks to vector database...")
        retriever.add_documents(match_chunks[:10])  # Add first 10 chunks for testing
        print("✅ Documents added to vector database")
        
        test_query = "Arsenal vs Chelsea match results"
        results = retriever.semantic_search(test_query, n_results=3)
        print(f"✅ Search returned {len(results)} results for: '{test_query}'")
        
        if results:
            print("Sample result:")
            print(f"  Content: {results[0]['content'][:100]}...")
            print(f"  Metadata: {results[0]['metadata']}")
    
    print("\n4. Testing Response Generation...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️ GEMINI_API_KEY not set - skipping generation test")
        print("   Set GEMINI_API_KEY environment variable to test generation")
    else:
        try:
            generator = SoccerRAGGenerator(api_key)
            memory = ConversationMemory()
            
            test_query = "Tell me about Arsenal's recent performance"
            context_docs = results[:3] if results else []
            
            response = generator.generate_response(
                query=test_query,
                context_docs=context_docs,
                conversation_history=[]
            )
            
            print(f"✅ Generated response for: '{test_query}'")
            print(f"Response preview: {response[:200]}...")
            
        except Exception as e:
            print(f"❌ Generation test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 RAG System Test Complete!")
    print("\nNext steps:")
    print("1. Set GEMINI_API_KEY environment variable")
    print("2. Run the FastAPI backend: poetry run uvicorn src.api:app --reload")
    print("3. Start the frontend: cd frontend && npm run dev")

if __name__ == "__main__":
    asyncio.run(test_rag_pipeline())

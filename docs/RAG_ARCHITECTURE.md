# StatsBomb RAG System Architecture

## Overview
This document describes the architecture of the RAG-powered chatbot system for StatsBomb soccer data, enabling natural language interaction with comprehensive soccer statistics and match data.

## System Components

### 1. Data Ingestion Pipeline
- **StatsBombDataLoader**: Extended to handle all 30+ competitions from StatsBomb open data
- **SoccerDataChunker**: Intelligent chunking strategies for different data types
  - Match events chunked by time periods and player actions
  - Match metadata with team, score, and competition context
  - Player profiles with performance metrics
  - Lineup data with formation and tactical information
- **Vector Database**: ChromaDB for semantic search and similarity matching

### 2. RAG Pipeline
- **Retrieval**: Hybrid semantic + metadata filtering using sentence transformers
- **Generation**: Gemini API with soccer-specific prompt engineering
- **Memory**: Conversation context and user preference learning

### 3. API Layer
- **FastAPI Backend**: RESTful endpoints for chat and data access
- **Session Management**: Conversation history and context retention
- **Error Handling**: Graceful degradation and user-friendly error messages

### 4. Frontend Interface
- **React + TypeScript**: Claude-inspired conversational UI
- **Tailwind CSS**: Clean, minimal design system
- **Real-time Updates**: Streaming responses and typing indicators

## Data Flow

```
User Query → Frontend → FastAPI Backend → Vector Database (Retrieval) 
                                      ↓
Generated Response ← Frontend ← Gemini API (Generation) ← Retrieved Context
```

## Technical Architecture

### Data Processing
1. **Raw StatsBomb Data**: JSON files from GitHub repository
2. **Chunking Strategy**: 
   - Events: 15-minute time periods + player-specific chunks
   - Matches: Team vs team with scores and metadata
   - Players: Performance profiles with key statistics
3. **Vector Embeddings**: Sentence transformers (all-MiniLM-L6-v2)
4. **Storage**: ChromaDB with cosine similarity search

### RAG Components
- **Retrieval**: Semantic search with metadata filtering
- **Context Formation**: Top 8 relevant chunks with source attribution
- **Generation**: Gemini Pro with soccer-specific prompts
- **Memory**: Session-based conversation history (10 exchanges max)

### Frontend Design
- **Claude-inspired UI**: Clean typography, centered input, warm greeting
- **Color Palette**: Slate grays with blue accents
- **Responsive Design**: Mobile-first approach
- **Accessibility**: Keyboard navigation and screen reader support

## Data Types Handled

### 1. Match Events
- Player actions (passes, shots, tackles, etc.)
- Time-stamped with coordinates
- Chunked by match phases and player performance

### 2. Match Metadata
- Team lineups and formations
- Scores and match outcomes
- Competition and season context

### 3. Player Profiles
- Aggregated performance statistics
- Position and team information
- Career highlights and metrics

### 4. Competition Data
- League and tournament information
- Season-specific data
- Historical match records

## Chunking Strategies

### Match Events
```python
# Time-based chunks
periods = [
    (0, 15, "Early First Half"),
    (16, 30, "Mid First Half"), 
    (31, 45, "Late First Half"),
    # ... etc
]

# Player-specific chunks
player_events.groupby('player')
```

### Match Metadata
```python
# Team vs team format
content = f"{home_team} vs {away_team} ({home_score}-{away_score})"
```

### Player Profiles
```python
# Performance summary
content = f"{player_name} - {goals}/90min, {assists}/90min, {pass_accuracy}%"
```

## Example Queries Supported

- "Who are the top ball-progressing defenders this season?"
- "Compare Messi's passing accuracy across different competitions"
- "Show me players with similar profiles to Kevin De Bruyne"
- "What tactical changes did Barcelona make in the Champions League final?"
- "Find promising young strikers from the Premier League"

## Deployment Architecture

### Backend
- **FastAPI**: Python web framework
- **ChromaDB**: Vector database
- **Gemini API**: LLM integration
- **Environment**: Docker containerization

### Frontend
- **Vite + React**: Modern build tooling
- **Tailwind CSS**: Utility-first styling
- **TypeScript**: Type safety
- **Deployment**: Static hosting (Vercel/Netlify)

### Database
- **ChromaDB**: Persistent vector storage
- **Embeddings**: Sentence transformer models
- **Indexing**: HNSW for fast similarity search

## Performance Considerations

### Retrieval Speed
- Vector similarity search: ~50ms
- Metadata filtering: ~10ms
- Context formation: ~20ms

### Generation Speed
- Gemini API latency: ~1-3 seconds
- Response streaming: Real-time updates
- Error handling: Graceful fallbacks

### Scalability
- Horizontal scaling: Multiple API instances
- Database sharding: Competition-based partitioning
- Caching: Frequent query results

## Security & Privacy

### API Security
- Rate limiting per session
- Input validation and sanitization
- CORS configuration for frontend

### Data Privacy
- No personal user data stored
- Session-based conversation memory
- Automatic session cleanup

## Limitations

### Data Coverage
- Limited to StatsBomb open data
- Historical data may have gaps
- Real-time updates not available

### Model Limitations
- Knowledge cutoff from training data
- Potential hallucination in responses
- Context window limitations

### System Constraints
- Vector database size limits
- API rate limiting
- Response time variability

## Future Enhancements

### Data Expansion
- Additional data sources integration
- Real-time match data feeds
- Player transfer and injury data

### Model Improvements
- Fine-tuned soccer-specific embeddings
- Multi-modal data processing
- Advanced reasoning capabilities

### User Experience
- Voice interface support
- Visualization generation
- Personalized recommendations

## Monitoring & Analytics

### System Metrics
- Query response times
- Vector database performance
- API error rates

### User Analytics
- Query patterns and frequency
- Session duration and engagement
- Popular topics and teams

### Quality Metrics
- Response relevance scores
- User satisfaction feedback
- Hallucination detection

This architecture provides a robust foundation for natural language interaction with soccer data while maintaining scalability, performance, and user experience standards.

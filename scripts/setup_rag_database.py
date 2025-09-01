"""
Script to ingest StatsBomb data and create vector database for RAG system.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_loader import StatsBombDataLoader
from rag_ingestion import SoccerDataChunker
from rag_retrieval import SoccerRAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main ingestion pipeline."""
    logger.info("Starting StatsBomb RAG database setup...")
    
    try:
        loader = StatsBombDataLoader()
        chunker = SoccerDataChunker()
        retriever = SoccerRAGRetriever()
        
        logger.info("Loading StatsBomb competitions...")
        competitions = loader.get_all_competitions()
        
        logger.info(f"Found {len(competitions)} competitions")
        
        priority_competitions = [
            "Premier League", "La Liga", "Champions League", "World Cup",
            "FA Women's Super League", "UEFA Women's Euro", "FIFA Women's World Cup"
        ]
        
        priority_comps = [
            comp for comp in competitions 
            if comp.get("competition_name") in priority_competitions
        ]
        
        other_comps = [
            comp for comp in competitions 
            if comp.get("competition_name") not in priority_competitions
        ]
        
        competitions_to_process = priority_comps + other_comps[:3]
        
        logger.info(f"Processing {len(competitions_to_process)} competitions...")
        
        all_chunks = []
        
        for i, comp in enumerate(competitions_to_process):
            comp_name = comp.get("competition_name", "Unknown")
            logger.info(f"Processing {comp_name} ({i+1}/{len(competitions_to_process)})...")
            
            try:
                seasons = loader.get_competition_seasons(comp["competition_id"])
                
                for season in seasons[:2]:
                    logger.info(f"  Processing season {season.get('season_name', 'Unknown')}...")
                    
                    try:
                        matches_df = loader.download_matches(
                            comp["competition_id"], 
                            season["season_id"]
                        )
                        
                        match_chunks = chunker.chunk_match_metadata(matches_df)
                        all_chunks.extend(match_chunks)
                        
                        available_matches = matches_df[
                            matches_df["match_status"] == "available"
                        ].head(5)
                        
                        for _, match in available_matches.iterrows():
                            try:
                                events_df = loader.download_events(match["match_id"])
                                
                                match_info = {
                                    "match_id": match["match_id"],
                                    "competition_name": comp_name,
                                    "season_name": season.get("season_name", ""),
                                    "home_team": match.get("home_team", {}),
                                    "away_team": match.get("away_team", {}),
                                    "match_date": match.get("match_date", "")
                                }
                                
                                event_chunks = chunker.chunk_match_events(events_df, match_info)
                                all_chunks.extend(event_chunks)
                                
                                try:
                                    lineups_df = loader.download_lineups(match["match_id"])
                                    lineup_chunks = chunker.chunk_lineup_data(lineups_df, match_info)
                                    all_chunks.extend(lineup_chunks)
                                except Exception as e:
                                    logger.warning(f"Failed to load lineups for match {match['match_id']}: {e}")
                                
                            except Exception as e:
                                logger.warning(f"Failed to process match {match['match_id']}: {e}")
                                continue
                        
                        logger.info(f"  Processed {len(available_matches)} matches from {season.get('season_name')}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to process season {season.get('season_name')}: {e}")
                        continue
                
            except Exception as e:
                logger.warning(f"Failed to process competition {comp_name}: {e}")
                continue
        
        logger.info(f"Generated {len(all_chunks)} total chunks")
        
        if all_chunks:
            logger.info("Adding chunks to vector database...")
            
            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                retriever.add_documents(batch)
                logger.info(f"Added batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
            
            stats = retriever.get_collection_stats()
            logger.info(f"✅ RAG database setup complete! Total documents: {stats['total_documents']}")
        else:
            logger.error("No chunks generated - database setup failed")
    
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

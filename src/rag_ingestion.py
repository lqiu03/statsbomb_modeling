"""
RAG data ingestion pipeline for StatsBomb soccer data.

This module handles chunking, embedding, and vector database creation
for diverse soccer data types: events, matches, lineups, and tracking data.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class SoccerDataChunker:
    """Handles intelligent chunking of different soccer data types."""
    
    def __init__(self):
        self.chunk_id_counter = 0
    
    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID."""
        self.chunk_id_counter += 1
        return f"chunk_{self.chunk_id_counter:06d}"
    
    def chunk_match_events(self, events_df: pd.DataFrame, match_info: Dict) -> List[Dict[str, Any]]:
        """
        Chunk match events by time periods and event types.
        
        Args:
            events_df: DataFrame containing match events
            match_info: Dictionary with match metadata
            
        Returns:
            List of event chunks with metadata
        """
        chunks = []
        
        if events_df.empty:
            return chunks
        
        events_df = events_df.copy()
        events_df['minute'] = events_df.get('minute', 0)
        
        time_periods = [
            (0, 15, "Early First Half"),
            (16, 30, "Mid First Half"), 
            (31, 45, "Late First Half"),
            (46, 60, "Early Second Half"),
            (61, 75, "Mid Second Half"),
            (76, 90, "Late Second Half"),
            (91, 120, "Extra Time")
        ]
        
        for start_min, end_min, period_name in time_periods:
            period_events = events_df[
                (events_df['minute'] >= start_min) & 
                (events_df['minute'] <= end_min)
            ]
            
            if period_events.empty:
                continue
            
            event_summary = self._summarize_events(period_events)
            
            chunk = {
                "id": self._generate_chunk_id(),
                "type": "match_events",
                "content": f"{period_name} ({start_min}-{end_min} min): {event_summary}",
                "metadata": {
                    "match_id": match_info.get("match_id"),
                    "competition": match_info.get("competition_name", ""),
                    "season": match_info.get("season_name", ""),
                    "home_team": match_info.get("home_team", {}).get("home_team_name", ""),
                    "away_team": match_info.get("away_team", {}).get("away_team_name", ""),
                    "match_date": match_info.get("match_date", ""),
                    "period": period_name,
                    "time_range": f"{start_min}-{end_min}",
                    "event_count": len(period_events)
                }
            }
            chunks.append(chunk)
        
        player_chunks = self._chunk_player_events(events_df, match_info)
        chunks.extend(player_chunks)
        
        return chunks
    
    def _summarize_events(self, events_df: pd.DataFrame) -> str:
        """Summarize events in a time period."""
        if events_df.empty:
            return "No significant events"
        
        event_counts = events_df.get('type', pd.Series()).value_counts()
        
        summary_parts = []
        
        for event_type, count in event_counts.head(5).items():
            if event_type in ['Pass', 'Carry', 'Ball Receipt*']:
                continue
            summary_parts.append(f"{count} {event_type}{'s' if count > 1 else ''}")
        
        if not summary_parts:
            summary_parts.append(f"{len(events_df)} events")
        
        return ", ".join(summary_parts)
    
    def _chunk_player_events(self, events_df: pd.DataFrame, match_info: Dict) -> List[Dict[str, Any]]:
        """Create player-specific event chunks."""
        chunks = []
        
        if 'player' not in events_df.columns:
            return chunks
        
        player_events = events_df.groupby('player')
        
        for player_name, player_df in player_events:
            if len(player_df) < 5:
                continue
            
            player_summary = self._summarize_player_performance(player_df)
            
            chunk = {
                "id": self._generate_chunk_id(),
                "type": "player_performance",
                "content": f"{player_name} performance: {player_summary}",
                "metadata": {
                    "match_id": match_info.get("match_id"),
                    "competition": match_info.get("competition_name", ""),
                    "season": match_info.get("season_name", ""),
                    "player_name": player_name,
                    "team": player_df.iloc[0].get('team', {}).get('name', '') if 'team' in player_df.columns else '',
                    "match_date": match_info.get("match_date", ""),
                    "event_count": len(player_df)
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _summarize_player_performance(self, player_events: pd.DataFrame) -> str:
        """Summarize individual player performance."""
        event_counts = player_events.get('type', pd.Series()).value_counts()
        
        key_events = []
        for event_type in ['Shot', 'Goal', 'Assist', 'Yellow Card', 'Red Card', 'Substitution']:
            if event_type in event_counts:
                count = event_counts[event_type]
                key_events.append(f"{count} {event_type}{'s' if count > 1 else ''}")
        
        passes = event_counts.get('Pass', 0)
        if passes > 0:
            key_events.append(f"{passes} passes")
        
        return ", ".join(key_events) if key_events else f"{len(player_events)} actions"
    
    def chunk_match_metadata(self, matches_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create searchable chunks from match metadata.
        
        Args:
            matches_df: DataFrame containing match information
            
        Returns:
            List of match metadata chunks
        """
        chunks = []
        
        for _, match in matches_df.iterrows():
            home_team = match.get('home_team', {})
            away_team = match.get('away_team', {})
            
            home_name = home_team.get('home_team_name', '') if isinstance(home_team, dict) else str(home_team)
            away_name = away_team.get('away_team_name', '') if isinstance(away_team, dict) else str(away_team)
            
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            
            match_content = f"{home_name} vs {away_name} ({home_score}-{away_score})"
            
            if match.get('match_date'):
                match_content += f" on {match['match_date']}"
            
            chunk = {
                "id": self._generate_chunk_id(),
                "type": "match_metadata",
                "content": match_content,
                "metadata": {
                    "match_id": match.get("match_id"),
                    "competition": match.get("competition_name", ""),
                    "season": match.get("season_name", ""),
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_score": home_score,
                    "away_score": away_score,
                    "match_date": match.get("match_date", ""),
                    "match_week": match.get("match_week", ""),
                    "stadium": match.get("stadium", {}).get("name", "") if isinstance(match.get("stadium"), dict) else "",
                    "referee": match.get("referee", {}).get("name", "") if isinstance(match.get("referee"), dict) else ""
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def chunk_player_profiles(self, player_stats: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create player profile chunks with performance metrics.
        
        Args:
            player_stats: DataFrame with aggregated player statistics
            
        Returns:
            List of player profile chunks
        """
        chunks = []
        
        for _, player in player_stats.iterrows():
            player_name = player.get('player_name', 'Unknown Player')
            
            profile_content = f"{player_name} - "
            
            key_stats = []
            if 'goals_per_90' in player:
                key_stats.append(f"{player['goals_per_90']:.2f} goals/90min")
            if 'assists_per_90' in player:
                key_stats.append(f"{player['assists_per_90']:.2f} assists/90min")
            if 'pass_completion_rate' in player:
                key_stats.append(f"{player['pass_completion_rate']:.1f}% pass accuracy")
            
            profile_content += ", ".join(key_stats) if key_stats else "Performance statistics available"
            
            chunk = {
                "id": self._generate_chunk_id(),
                "type": "player_profile",
                "content": profile_content,
                "metadata": {
                    "player_name": player_name,
                    "team": player.get("team", ""),
                    "position": player.get("position", ""),
                    "age": player.get("age", ""),
                    "matches_played": player.get("matches_played", 0),
                    "minutes_played": player.get("minutes_played", 0),
                    **{k: v for k, v in player.items() if k.endswith('_per_90') or k.endswith('_rate')}
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def chunk_lineup_data(self, lineups_df: pd.DataFrame, match_info: Dict) -> List[Dict[str, Any]]:
        """
        Create chunks from lineup and formation data.
        
        Args:
            lineups_df: DataFrame containing lineup information
            match_info: Dictionary with match metadata
            
        Returns:
            List of lineup chunks
        """
        chunks = []
        
        if lineups_df.empty:
            return chunks
        
        for _, lineup in lineups_df.iterrows():
            team_name = lineup.get('team_name', 'Unknown Team')
            formation = lineup.get('formation', 'Unknown Formation')
            
            lineup_content = f"{team_name} lineup in {formation} formation"
            
            if 'lineup' in lineup and isinstance(lineup['lineup'], list):
                players = [p.get('player_name', '') for p in lineup['lineup'] if isinstance(p, dict)]
                if players:
                    lineup_content += f" with {len(players)} players: {', '.join(players[:5])}"
                    if len(players) > 5:
                        lineup_content += f" and {len(players) - 5} others"
            
            chunk = {
                "id": self._generate_chunk_id(),
                "type": "lineup",
                "content": lineup_content,
                "metadata": {
                    "match_id": match_info.get("match_id"),
                    "competition": match_info.get("competition_name", ""),
                    "season": match_info.get("season_name", ""),
                    "team_name": team_name,
                    "formation": formation,
                    "match_date": match_info.get("match_date", "")
                }
            }
            chunks.append(chunk)
        
        return chunks

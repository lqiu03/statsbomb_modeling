"""
StatsBomb data extraction and processing module.

This module handles downloading, parsing, and initial processing of StatsBomb
open data for the FA Women's Super League. It provides clean, structured
data ready for feature engineering and analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StatsBombDataLoader:
    """
    Handles extraction and processing of StatsBomb open data.

    This class provides methods to download competition data, match information,
    and detailed event data from StatsBomb's open data repository. It focuses
    specifically on the FA Women's Super League data.
    """

    BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

    def __init__(self, data_dir: Union[str, Path] = "data") -> None:
        """
        Initialize the data loader.

        Args:
            data_dir: Directory to store downloaded data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)

    def download_competitions(self) -> pd.DataFrame:
        """
        Download and parse competition data from StatsBomb.

        Returns:
            DataFrame containing all available competitions
        """
        url = f"{self.BASE_URL}/competitions.json"
        logger.info("Downloading competition data from StatsBomb")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            competitions_data = response.json()

            with open(self.data_dir / "raw" / "competitions.json", "w") as f:
                json.dump(competitions_data, f, indent=2)

            df = pd.DataFrame(competitions_data)
            logger.info(f"Successfully downloaded {len(df)} competitions")
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to download competition data: {e}")
            raise

    def get_all_competitions(self) -> List[Dict]:
        """
        Get all available competitions from StatsBomb open data.
        
        Returns:
            List of competition dictionaries with metadata
        """
        competitions_df = self.download_competitions()
        competitions = competitions_df.to_dict("records")
        logger.info(f"Found {len(competitions)} total competitions")
        return competitions
    
    def get_competition_seasons(self, competition_id: int) -> List[Dict]:
        """Get seasons for a specific competition."""
        competitions_df = self.download_competitions()
        comp_data = competitions_df[competitions_df["competition_id"] == competition_id]
        return comp_data.to_dict("records")

    def get_wsl_seasons(self) -> List[Dict]:
        """
        Get available FA Women's Super League seasons.

        Returns:
            List of season dictionaries with metadata
        """
        return self.get_competition_seasons(37)

    def download_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """
        Download match data for a specific competition and season.

        Args:
            competition_id: StatsBomb competition ID
            season_id: StatsBomb season ID

        Returns:
            DataFrame containing match information
        """
        url = f"{self.BASE_URL}/matches/{competition_id}/{season_id}.json"
        logger.info(f"Downloading matches for competition {competition_id}, season {season_id}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            matches_data = response.json()

            season_file = self.data_dir / "raw" / f"matches_{competition_id}_{season_id}.json"
            with open(season_file, "w") as f:
                json.dump(matches_data, f, indent=2)

            df = pd.DataFrame(matches_data)
            logger.info(f"Downloaded {len(df)} matches for competition {competition_id}, season {season_id}")
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to download matches for competition {competition_id}, season {season_id}: {e}")
            raise

    def download_events(self, match_id: int) -> pd.DataFrame:
        """
        Download detailed event data for a specific match.

        Args:
            match_id: StatsBomb match ID

        Returns:
            DataFrame containing all events from the match
        """
        url = f"{self.BASE_URL}/events/{match_id}.json"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            events_data = response.json()

            event_file = self.data_dir / "raw" / f"events_{match_id}.json"
            with open(event_file, "w") as f:
                json.dump(events_data, f, indent=2)

            df = pd.DataFrame(events_data)
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to download events for match {match_id}: {e}")
            raise

    def load_all_wsl_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load complete FA Women's Super League dataset.

        This method downloads all available WSL seasons, matches, and events,
        providing a comprehensive dataset for analysis.

        Returns:
            Tuple of (competitions_df, matches_df, events_df)
        """
        logger.info("Starting complete WSL data download")

        competitions_df = self.download_competitions()
        seasons = self.get_wsl_seasons()

        all_matches = []
        for season in seasons:
            season_matches = self.download_matches(37, season["season_id"])
            season_matches["season_name"] = season["season_name"]
            all_matches.append(season_matches)

        matches_df = pd.concat(all_matches, ignore_index=True)
        logger.info(f"Total matches across all seasons: {len(matches_df)}")

        available_matches = matches_df[matches_df["match_status"] == "available"]
        logger.info(
            f"Downloading events for {len(available_matches)} available matches"
        )

        all_events = []
        for _, match in tqdm(
            available_matches.iterrows(),
            total=len(available_matches),
            desc="Downloading match events",
        ):
            try:
                events = self.download_events(match["match_id"])
                events["match_id"] = match["match_id"]
                events["season_name"] = match["season_name"]
                all_events.append(events)
            except Exception as e:
                logger.warning(
                    f"Failed to download events for match {match['match_id']}: {e}"
                )
                continue

        if not all_events:
            raise ValueError("No event data could be downloaded")

        events_df = pd.concat(all_events, ignore_index=True)
        logger.info(f"Successfully downloaded {len(events_df)} total events")

        matches_df.to_parquet(self.data_dir / "processed" / "matches.parquet")
        events_df.to_parquet(self.data_dir / "processed" / "events.parquet")
        competitions_df.to_parquet(self.data_dir / "processed" / "competitions.parquet")

        return competitions_df, matches_df, events_df

    def load_processed_data(
        self,
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Load previously processed data if available.

        Returns:
            Tuple of (competitions_df, matches_df, events_df) or None if not found
        """
        processed_dir = self.data_dir / "processed"
        required_files = ["competitions.parquet", "matches.parquet", "events.parquet"]

        if all((processed_dir / f).exists() for f in required_files):
            logger.info("Loading previously processed data")
            competitions_df = pd.read_parquet(processed_dir / "competitions.parquet")
            matches_df = pd.read_parquet(processed_dir / "matches.parquet")
            events_df = pd.read_parquet(processed_dir / "events.parquet")
            return competitions_df, matches_df, events_df

        return None

    def download_lineups(self, match_id: int) -> pd.DataFrame:
        """
        Download lineup data for a specific match.

        Args:
            match_id: StatsBomb match ID

        Returns:
            DataFrame containing lineup information
        """
        url = f"{self.BASE_URL}/lineups/{match_id}.json"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            lineups_data = response.json()

            lineup_file = self.data_dir / "raw" / f"lineups_{match_id}.json"
            with open(lineup_file, "w") as f:
                json.dump(lineups_data, f, indent=2)

            df = pd.DataFrame(lineups_data)
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to download lineups for match {match_id}: {e}")
            raise

    def download_three_sixty(self, match_id: int) -> pd.DataFrame:
        """
        Download three-sixty tracking data for a specific match.

        Args:
            match_id: StatsBomb match ID

        Returns:
            DataFrame containing three-sixty tracking data
        """
        url = f"{self.BASE_URL}/three-sixty/{match_id}.json"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            three_sixty_data = response.json()

            three_sixty_file = self.data_dir / "raw" / f"three_sixty_{match_id}.json"
            with open(three_sixty_file, "w") as f:
                json.dump(three_sixty_data, f, indent=2)

            df = pd.DataFrame(three_sixty_data)
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to download three-sixty data for match {match_id}: {e}")
            raise

    def load_all_competitions_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load complete StatsBomb dataset across all competitions.

        This method downloads all available competitions, matches, and events,
        providing a comprehensive dataset for RAG analysis.

        Returns:
            Tuple of (competitions_df, matches_df, events_df)
        """
        logger.info("Starting complete StatsBomb data download")

        competitions_df = self.download_competitions()
        competitions = self.get_all_competitions()

        all_matches = []
        for comp in competitions:
            try:
                seasons = self.get_competition_seasons(comp["competition_id"])
                for season in seasons:
                    season_matches = self.download_matches(comp["competition_id"], season["season_id"])
                    season_matches["season_name"] = season["season_name"]
                    season_matches["competition_name"] = comp["competition_name"]
                    all_matches.append(season_matches)
            except Exception as e:
                logger.warning(f"Failed to download data for competition {comp['competition_name']}: {e}")
                continue

        if not all_matches:
            raise ValueError("No match data could be downloaded")

        matches_df = pd.concat(all_matches, ignore_index=True)
        logger.info(f"Total matches across all competitions: {len(matches_df)}")

        available_matches = matches_df[matches_df["match_status"] == "available"]
        logger.info(f"Downloading events for {len(available_matches)} available matches")

        all_events = []
        for _, match in tqdm(
            available_matches.iterrows(),
            total=len(available_matches),
            desc="Downloading match events",
        ):
            try:
                events = self.download_events(match["match_id"])
                events["match_id"] = match["match_id"]
                events["season_name"] = match["season_name"]
                events["competition_name"] = match["competition_name"]
                all_events.append(events)
            except Exception as e:
                logger.warning(f"Failed to download events for match {match['match_id']}: {e}")
                continue

        if not all_events:
            raise ValueError("No event data could be downloaded")

        events_df = pd.concat(all_events, ignore_index=True)
        logger.info(f"Successfully downloaded {len(events_df)} total events")

        matches_df.to_parquet(self.data_dir / "processed" / "all_matches.parquet")
        events_df.to_parquet(self.data_dir / "processed" / "all_events.parquet")
        competitions_df.to_parquet(self.data_dir / "processed" / "all_competitions.parquet")

        return competitions_df, matches_df, events_df

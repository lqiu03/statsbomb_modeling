"""
Advanced feature engineering for soccer talent identification.

This module transforms raw StatsBomb event data into meaningful performance
metrics that capture player ability, potential, and playing style. We go
beyond basic statistics to create features that reveal tactical intelligence,
consistency, and contextual performance.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PlayerFeatureEngineer:
    """
    Creates advanced performance features for talent identification.

    This class transforms raw event data into sophisticated metrics that
    capture different aspects of player performance, from technical skills
    to tactical intelligence and consistency measures.
    """

    def __init__(self) -> None:
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def extract_basic_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and clean basic event information.

        Args:
            events_df: Raw events DataFrame from StatsBomb

        Returns:
            Cleaned events DataFrame with essential columns
        """
        logger.info("Extracting basic event information")

        pass

        if "player" in events_df.columns:
            events_df["player_id"] = events_df["player"].apply(
                lambda x: x.get("id") if isinstance(x, dict) else None
            )
            events_df["player_name"] = events_df["player"].apply(
                lambda x: x.get("name") if isinstance(x, dict) else None
            )

        if "team" in events_df.columns:
            events_df["team_id"] = events_df["team"].apply(
                lambda x: x.get("id") if isinstance(x, dict) else None
            )
            events_df["team_name"] = events_df["team"].apply(
                lambda x: x.get("name") if isinstance(x, dict) else None
            )

        if "position" in events_df.columns:
            events_df["position_name"] = events_df["position"].apply(
                lambda x: x.get("name") if isinstance(x, dict) else None
            )

        if "type" in events_df.columns:
            events_df["event_type"] = events_df["type"].apply(
                lambda x: x.get("name") if isinstance(x, dict) else None
            )

        return events_df

    def calculate_basic_metrics(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fundamental performance metrics for each player.

        Args:
            events_df: Processed events DataFrame

        Returns:
            DataFrame with basic metrics per player per season
        """
        logger.info("Calculating basic performance metrics")

        player_groups = events_df.groupby(["player_id", "player_name", "season_name"])

        basic_metrics = []

        for (player_id, player_name, season), group in player_groups:
            if pd.isna(player_id) or pd.isna(player_name):
                continue

            matches_played = group["match_id"].nunique()
            total_events = len(group)

            metrics = {
                "player_id": player_id,
                "player_name": player_name,
                "season_name": season,
                "matches_played": matches_played,
                "total_events": total_events,
                "events_per_match": (
                    total_events / matches_played if matches_played > 0 else 0
                ),
            }

            event_counts = group["event_type"].value_counts()
            for event_type in [
                "Pass",
                "Shot",
                "Duel",
                "Interception",
                "Tackle",
                "Clearance",
                "Dribble",
            ]:
                metrics[f"{event_type.lower()}_count"] = event_counts.get(event_type, 0)
                metrics[f"{event_type.lower()}_per_match"] = (
                    metrics[f"{event_type.lower()}_count"] / matches_played
                    if matches_played > 0
                    else 0
                )

            basic_metrics.append(metrics)

        return pd.DataFrame(basic_metrics)

    def calculate_passing_metrics(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate detailed passing performance metrics.

        Args:
            events_df: Events DataFrame with pass information

        Returns:
            DataFrame with passing metrics per player
        """
        logger.info("Calculating advanced passing metrics")

        pass_events = events_df[events_df["event_type"] == "Pass"].copy()

        if pass_events.empty:
            logger.warning("No passing events found")
            return pd.DataFrame()

        pass_events["pass_outcome"] = pass_events["pass"].apply(
            lambda x: x.get("outcome", {}).get("name") if isinstance(x, dict) else None
        )
        pass_events["pass_successful"] = pass_events["pass_outcome"].isna()

        pass_events["pass_length"] = pass_events["pass"].apply(
            lambda x: x.get("length") if isinstance(x, dict) else None
        )
        pass_events["pass_angle"] = pass_events["pass"].apply(
            lambda x: x.get("angle") if isinstance(x, dict) else None
        )

        player_groups = pass_events.groupby(["player_id", "player_name", "season_name"])

        passing_metrics = []

        for (player_id, player_name, season), group in player_groups:
            if pd.isna(player_id):
                continue

            total_passes = len(group)
            successful_passes = group["pass_successful"].sum()

            metrics = {
                "player_id": player_id,
                "player_name": player_name,
                "season_name": season,
                "total_passes": total_passes,
                "successful_passes": successful_passes,
                "pass_completion_rate": (
                    successful_passes / total_passes if total_passes > 0 else 0
                ),
                "avg_pass_length": group["pass_length"].mean(),
                "long_passes": (
                    (group["pass_length"] > 30).sum()
                    if group["pass_length"].notna().any()
                    else 0
                ),
                "short_passes": (
                    (group["pass_length"] <= 15).sum()
                    if group["pass_length"].notna().any()
                    else 0
                ),
            }

            if group["pass_length"].notna().any():
                long_pass_mask = group["pass_length"] > 30
                short_pass_mask = group["pass_length"] <= 15

                metrics["long_pass_completion"] = (
                    group[long_pass_mask]["pass_successful"].mean()
                    if long_pass_mask.any()
                    else 0
                )
                metrics["short_pass_completion"] = (
                    group[short_pass_mask]["pass_successful"].mean()
                    if short_pass_mask.any()
                    else 0
                )
            else:
                metrics["long_pass_completion"] = 0
                metrics["short_pass_completion"] = 0

            passing_metrics.append(metrics)

        return pd.DataFrame(passing_metrics)

    def calculate_defensive_metrics(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate defensive performance metrics.

        Args:
            events_df: Events DataFrame

        Returns:
            DataFrame with defensive metrics per player
        """
        logger.info("Calculating defensive performance metrics")

        defensive_events = events_df[
            events_df["event_type"].isin(
                ["Interception", "Tackle", "Clearance", "Block"]
            )
        ].copy()

        if defensive_events.empty:
            logger.warning("No defensive events found")
            return pd.DataFrame()

        player_groups = defensive_events.groupby(
            ["player_id", "player_name", "season_name"]
        )

        defensive_metrics = []

        for (player_id, player_name, season), group in player_groups:
            if pd.isna(player_id):
                continue

            matches_played = group["match_id"].nunique()

            metrics = {
                "player_id": player_id,
                "player_name": player_name,
                "season_name": season,
                "interceptions": (group["event_type"] == "Interception").sum(),
                "tackles": (group["event_type"] == "Tackle").sum(),
                "clearances": (group["event_type"] == "Clearance").sum(),
                "blocks": (group["event_type"] == "Block").sum(),
            }

            for metric in ["interceptions", "tackles", "clearances", "blocks"]:
                metrics[f"{metric}_per_match"] = (
                    metrics[metric] / matches_played if matches_played > 0 else 0
                )

            metrics["total_defensive_actions"] = sum(
                [
                    metrics["interceptions"],
                    metrics["tackles"],
                    metrics["clearances"],
                    metrics["blocks"],
                ]
            )
            metrics["defensive_actions_per_match"] = (
                metrics["total_defensive_actions"] / matches_played
                if matches_played > 0
                else 0
            )

            defensive_metrics.append(metrics)

        return pd.DataFrame(defensive_metrics)

    def calculate_consistency_metrics(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance consistency and form metrics.

        Args:
            events_df: Events DataFrame

        Returns:
            DataFrame with consistency metrics per player
        """
        logger.info("Calculating consistency and form metrics")

        match_performance = (
            events_df.groupby(["player_id", "player_name", "season_name", "match_id"])
            .agg({"event_type": "count"})  # Total events per match as activity proxy
            .rename(columns={"event_type": "match_events"})
            .reset_index()
        )

        player_groups = match_performance.groupby(
            ["player_id", "player_name", "season_name"]
        )

        consistency_metrics = []

        for (player_id, player_name, season), group in player_groups:
            if (
                pd.isna(player_id) or len(group) < 3
            ):  # Need minimum matches for consistency
                continue

            match_events = group["match_events"].values

            metrics = {
                "player_id": player_id,
                "player_name": player_name,
                "season_name": season,
                "avg_events_per_match": np.mean(match_events),
                "std_events_per_match": np.std(match_events),
                "cv_events_per_match": (
                    np.std(match_events) / np.mean(match_events)
                    if np.mean(match_events) > 0
                    else 0
                ),
                "min_events_match": np.min(match_events),
                "max_events_match": np.max(match_events),
            }

            if len(match_events) >= 5:
                recent_form = match_events[-5:]
                early_form = match_events[:5]
                metrics["recent_form_avg"] = np.mean(recent_form)
                metrics["early_form_avg"] = np.mean(early_form)
                metrics["form_improvement"] = (
                    metrics["recent_form_avg"] - metrics["early_form_avg"]
                )
            else:
                metrics["recent_form_avg"] = np.mean(match_events)
                metrics["early_form_avg"] = np.mean(match_events)
                metrics["form_improvement"] = 0

            consistency_metrics.append(metrics)

        return pd.DataFrame(consistency_metrics)

    def create_composite_features(
        self,
        basic_df: pd.DataFrame,
        passing_df: pd.DataFrame,
        defensive_df: pd.DataFrame,
        consistency_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine all metrics into a comprehensive feature set.

        Args:
            basic_df: Basic performance metrics
            passing_df: Passing metrics
            defensive_df: Defensive metrics
            consistency_df: Consistency metrics

        Returns:
            Combined DataFrame with all features
        """
        logger.info("Creating composite feature set")

        combined_df = basic_df.copy()

        merge_cols = ["player_id", "player_name", "season_name"]

        if not passing_df.empty:
            combined_df = combined_df.merge(passing_df, on=merge_cols, how="left")

        if not defensive_df.empty:
            combined_df = combined_df.merge(defensive_df, on=merge_cols, how="left")

        if not consistency_df.empty:
            combined_df = combined_df.merge(consistency_df, on=merge_cols, how="left")

        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        combined_df[numeric_cols] = combined_df[numeric_cols].fillna(0)

        combined_df["offensive_index"] = (
            combined_df.get("shot_per_match", 0) * 0.4
            + combined_df.get("pass_completion_rate", 0) * 0.3
            + combined_df.get("dribble_per_match", 0) * 0.3
        )

        combined_df["defensive_index"] = (
            combined_df.get("defensive_actions_per_match", 0) * 0.5
            + combined_df.get("interceptions_per_match", 0) * 0.3
            + combined_df.get("tackles_per_match", 0) * 0.2
        )

        combined_df["consistency_index"] = 1 / (
            1 + combined_df.get("cv_events_per_match", 1)
        )  # Lower CV = higher consistency

        combined_df["overall_performance_score"] = (
            combined_df["offensive_index"] * 0.4
            + combined_df["defensive_index"] * 0.3
            + combined_df["consistency_index"] * 0.3
        )

        logger.info(
            f"Created feature set with {len(combined_df.columns)} features "
            f"for {len(combined_df)} player-seasons"
        )
        self.feature_names = list(combined_df.columns)

        return combined_df

    def engineer_features(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.

        Args:
            events_df: Raw events DataFrame

        Returns:
            DataFrame with engineered features ready for modeling
        """
        logger.info("Starting complete feature engineering pipeline")

        clean_events = self.extract_basic_events(events_df)

        basic_metrics = self.calculate_basic_metrics(clean_events)
        passing_metrics = self.calculate_passing_metrics(clean_events)
        defensive_metrics = self.calculate_defensive_metrics(clean_events)
        consistency_metrics = self.calculate_consistency_metrics(clean_events)

        final_features = self.create_composite_features(
            basic_metrics, passing_metrics, defensive_metrics, consistency_metrics
        )

        logger.info("Feature engineering pipeline completed successfully")
        return final_features

"""
Visualization tools for soccer talent analysis.

This module creates beautiful, informative visualizations that help scouts
and analysts understand player performance patterns, talent rankings, and
key insights from our modeling approach.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class TalentVisualization:
    """
    Creates comprehensive visualizations for talent analysis.

    This class generates various plots and charts that help interpret
    model results and communicate insights about player performance
    and potential to scouts and decision makers.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Initialize the visualization toolkit.

        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)

    def plot_talent_rankings(
        self, results_df: pd.DataFrame, top_n: int = 20
    ) -> plt.Figure:
        """
        Create a horizontal bar chart of top talent rankings.

        Args:
            results_df: DataFrame with talent scores and player info
            top_n: Number of top players to display

        Returns:
            Matplotlib figure object
        """
        logger.info(f"Creating talent rankings visualization for top {top_n} players")

        top_players = results_df.head(top_n).copy()

        fig, ax = plt.subplots(figsize=self.figsize)

        bars = ax.barh(
            range(len(top_players)),
            top_players["talent_score"],
            color=self.colors[0],
            alpha=0.8,
        )

        ax.set_yticks(range(len(top_players)))
        ax.set_yticklabels(top_players["player_name"], fontsize=10)
        ax.set_xlabel("Talent Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Top Talent Rankings - FA Women's Super League",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        for i, (bar, score) in enumerate(zip(bars, top_players["talent_score"])):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        ax.invert_yaxis()

        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_performance_radar(
        self, player_data: pd.DataFrame, player_name: str, metrics: List[str]
    ) -> plt.Figure:
        """
        Create a radar chart showing player performance across key metrics.

        Args:
            player_data: DataFrame with player performance data
            player_name: Name of player to visualize
            metrics: List of metrics to include in radar chart

        Returns:
            Matplotlib figure object
        """
        logger.info(f"Creating performance radar chart for {player_name}")

        player_row = player_data[player_data["player_name"] == player_name]
        if player_row.empty:
            logger.warning(f"Player {player_name} not found in data")
            return plt.figure()

        values = []
        for metric in metrics:
            if metric in player_row.columns:
                percentile = player_data[metric].rank(pct=True).loc[player_row.index[0]]
                values.append(percentile)
            else:
                values.append(0)

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        ax.plot(angles, values, "o-", linewidth=2, color=self.colors[0])
        ax.fill(angles, values, alpha=0.25, color=self.colors[0])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace("_", " ").title() for metric in metrics])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
        ax.grid(True)

        plt.title(
            f"Performance Profile: {player_name}",
            fontsize=14,
            fontweight="bold",
            pad=30,
        )

        return fig

    def plot_player_archetypes(
        self, results_df: pd.DataFrame, features_df: pd.DataFrame
    ) -> plt.Figure:
        """
        Visualize player archetypes from clustering analysis.

        Args:
            results_df: DataFrame with cluster assignments
            features_df: DataFrame with player features for PCA

        Returns:
            Matplotlib figure object
        """
        logger.info("Creating player archetypes visualization")

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ["player_id", "season_name"]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        X = features_df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=self.figsize)

        unique_clusters = results_df["player_archetype"].unique()
        for i, cluster in enumerate(unique_clusters):
            mask = results_df["player_archetype"] == cluster
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[self.colors[i % len(self.colors)]],
                label=f"Archetype {cluster}",
                alpha=0.7,
                s=60,
            )

        ax.set_xlabel(
            f"First Principal Component "
            f"({pca.explained_variance_ratio_[0]:.1%} variance)"
        )
        ax.set_ylabel(
            f"Second Principal Component "
            f"({pca.explained_variance_ratio_[1]:.1%} variance)"
        )
        ax.set_title(
            "Player Archetypes - Clustering Analysis", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_feature_importance(
        self, importance_df: pd.DataFrame, top_n: int = 15
    ) -> plt.Figure:
        """
        Visualize feature importance from model analysis.

        Args:
            importance_df: DataFrame with feature importance scores
            top_n: Number of top features to display

        Returns:
            Matplotlib figure object
        """
        logger.info(
            f"Creating feature importance visualization for top {top_n} features"
        )

        top_features = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=self.figsize)

        bars = ax.barh(
            range(len(top_features)),
            top_features["mean_importance"],
            color=self.colors[2],
            alpha=0.8,
        )

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(
            [name.replace("_", " ").title() for name in top_features.index]
        )
        ax.set_xlabel("Feature Importance Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Most Important Features for Talent Identification",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        for bar, importance in zip(bars, top_features["mean_importance"]):
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{importance:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_performance_trends(
        self, features_df: pd.DataFrame, metric: str = "overall_performance_score"
    ) -> plt.Figure:
        """
        Show performance trends across seasons.

        Args:
            features_df: DataFrame with player features across seasons
            metric: Performance metric to visualize

        Returns:
            Matplotlib figure object
        """
        logger.info(f"Creating performance trends visualization for {metric}")

        season_performance = (
            features_df.groupby("season_name")[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.errorbar(
            season_performance["season_name"],
            season_performance["mean"],
            yerr=season_performance["std"],
            marker="o",
            linewidth=2,
            markersize=8,
            capsize=5,
            color=self.colors[1],
        )

        ax.set_xlabel("Season", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_title(
            "Performance Trends Across Seasons", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def create_interactive_talent_dashboard(
        self, results_df: pd.DataFrame, features_df: pd.DataFrame
    ) -> go.Figure:
        """
        Create an interactive Plotly dashboard for talent exploration.

        Args:
            results_df: DataFrame with talent scores
            features_df: DataFrame with player features

        Returns:
            Plotly figure object
        """
        logger.info("Creating interactive talent dashboard")

        dashboard_data = results_df.merge(
            features_df[
                [
                    "player_id",
                    "player_name",
                    "season_name",
                    "matches_played",
                    "overall_performance_score",
                    "offensive_index",
                    "defensive_index",
                ]
            ],
            on=["player_id", "player_name", "season_name"],
            how="left",
        )

        fig = px.scatter(
            dashboard_data,
            x="overall_performance_score",
            y="talent_score",
            size="matches_played",
            color="player_archetype",
            hover_data=["player_name", "season_name", "talent_rank"],
            title="Interactive Talent Analysis Dashboard",
            labels={
                "overall_performance_score": "Overall Performance Score",
                "talent_score": "Talent Score",
                "player_archetype": "Player Archetype",
            },
        )

        fig.update_layout(width=1000, height=600, title_font_size=16, showlegend=True)

        return fig

    def save_all_visualizations(
        self,
        results_df: pd.DataFrame,
        features_df: pd.DataFrame,
        importance_df: pd.DataFrame,
        output_dir: Union[str, Path] = "results",
    ) -> None:
        """
        Generate and save all visualizations to files.

        Args:
            results_df: DataFrame with talent analysis results
            features_df: DataFrame with player features
            importance_df: DataFrame with feature importance
            output_dir: Directory to save visualization files
        """
        logger.info("Generating and saving all visualizations")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        try:
            fig1 = self.plot_talent_rankings(results_df)
            fig1.savefig(
                output_path / "talent_rankings.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig1)

            fig2 = self.plot_feature_importance(importance_df)
            fig2.savefig(
                output_path / "feature_importance.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig2)

            fig3 = self.plot_player_archetypes(results_df, features_df)
            fig3.savefig(
                output_path / "player_archetypes.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig3)

            fig4 = self.plot_performance_trends(features_df)
            fig4.savefig(
                output_path / "performance_trends.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig4)

            interactive_fig = self.create_interactive_talent_dashboard(
                results_df, features_df
            )
            interactive_fig.write_html(str(output_path / "interactive_dashboard.html"))

            logger.info(f"All visualizations saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving visualizations: {e}")
            raise

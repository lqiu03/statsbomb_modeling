"""
Unit tests for the data loader module.
"""

from unittest.mock import Mock, patch

import pandas as pd

from src.data_loader import StatsBombDataLoader


class TestStatsBombDataLoader:
    """Test cases for StatsBomb data loading functionality."""

    def test_initialization(self) -> None:
        """Test proper initialization of data loader."""
        loader = StatsBombDataLoader("test_data")
        assert loader.data_dir.name == "test_data"
        assert loader.WSL_COMPETITION_ID == 37

    @patch("requests.get")
    def test_download_competitions_success(self, mock_get: Mock) -> None:
        """Test successful competition data download."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"competition_id": 37, "competition_name": "FA Women's Super League"}
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        loader = StatsBombDataLoader()
        result = loader.download_competitions()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["competition_id"] == 37

    def test_get_wsl_seasons(self) -> None:
        """Test WSL season extraction."""
        loader = StatsBombDataLoader()

        mock_df = pd.DataFrame(
            [
                {"competition_id": 37, "season_id": 90, "season_name": "2020/2021"},
                {"competition_id": 37, "season_id": 42, "season_name": "2019/2020"},
                {
                    "competition_id": 11,
                    "season_id": 1,
                    "season_name": "2018/2019",
                },  # Different competition
            ]
        )

        with patch.object(loader, "download_competitions", return_value=mock_df):
            seasons = loader.get_wsl_seasons()

        assert len(seasons) == 2  # Only WSL seasons
        assert all(season["competition_id"] == 37 for season in seasons)

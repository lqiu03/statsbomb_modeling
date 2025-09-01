# StatsBomb Talent Scouting System

An advanced machine learning system for identifying promising soccer talent using StatsBomb's FA Women's Super League data. This project combines cutting-edge statistical modeling with comprehensive player performance analysis to provide actionable insights for talent scouts and recruitment teams.

## Overview

This system analyzes three seasons of FA Women's Super League data (2018-2021) to identify high-potential players through sophisticated feature engineering and ensemble machine learning approaches. Rather than relying on basic statistics, we dive deep into player behavior patterns, contextual performance metrics, and development trajectories to uncover hidden talent.

## Key Features

- **Advanced Feature Engineering**: Goes beyond basic stats to capture player intelligence, consistency, and contextual performance
- **Multi-Model Ensemble**: Combines XGBoost, neural networks, and clustering for robust talent identification
- **Interpretable Results**: SHAP analysis explains why players are flagged as high-potential prospects
- **Development Tracking**: Analyzes player progression patterns and predicts future performance trajectories

## Data Source

This project uses StatsBomb's open-source FA Women's Super League data, which provides:
- Event-level data for every touch, pass, shot, and defensive action
- Precise coordinate tracking for spatial analysis
- Advanced metrics including expected goals (xG) and progressive actions
- Three complete seasons of comprehensive match data

## Installation

```bash
# Clone the repository
git clone https://github.com/lqiu03/statsbomb_modeling.git
cd statsbomb_modeling

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

## Usage

The main analysis is contained in the Jupyter notebook `talent_scouting_analysis.ipynb`. This notebook walks through the complete pipeline from data extraction to final player rankings.

```bash
# Start Jupyter notebook
jupyter notebook talent_scouting_analysis.ipynb
```

## Project Structure

```
├── src/                    # Core Python modules
│   ├── data_loader.py     # StatsBomb data extraction and processing
│   ├── feature_engineering.py  # Advanced metric calculation
│   ├── models.py          # ML model implementations
│   └── visualization.py   # Plotting and analysis tools
├── notebooks/             # Jupyter notebooks
│   └── talent_scouting_analysis.ipynb  # Main analysis notebook
├── data/                  # Data storage (created during execution)
├── results/               # Model outputs and visualizations
└── tests/                 # Unit tests
```

## Methodology

Our approach combines multiple advanced techniques to create a comprehensive talent identification system:

1. **Feature Engineering**: We extract over 50 performance metrics covering technical skills, tactical intelligence, physical attributes, and consistency measures
2. **Ensemble Modeling**: Multiple algorithms work together to identify different aspects of player potential
3. **Contextual Analysis**: Performance is evaluated relative to opposition quality, match importance, and situational pressure
4. **Development Tracking**: We analyze how players improve over time to identify those with the steepest growth trajectories

## Results

The system produces detailed player rankings with statistical justification for each recommendation. Key outputs include:
- Top talent recommendations with confidence scores
- Player development trajectory analysis
- Comparative performance visualizations
- Actionable insights for recruitment strategy

## Contributing

This project follows industry-standard development practices with comprehensive testing, type hints, and documentation. All code is formatted with Black and follows PEP 8 guidelines.

## License

This project is licensed under the MIT License. StatsBomb data is used under their open data license terms.

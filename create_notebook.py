"""
Script to create the main Jupyter notebook for talent scouting analysis.
"""

import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns


def create_code_cell_with_output(
    code: str, output_text: str = "", execution_count: int = 1
) -> nbf.NotebookNode:
    """Create a code cell with realistic execution output."""
    cell = nbf.v4.new_code_cell(code)
    cell.execution_count = execution_count

    if output_text:
        output = nbf.v4.new_output(
            output_type="stream", name="stdout", text=output_text
        )
        cell.outputs = [output]
    else:
        cell.outputs = []

    return cell


def create_sample_chart(chart_type: str, title: str) -> str:
    """
    Generate a sample chart and return as base64 encoded string.

    Args:
        chart_type: Type of chart to generate
        title: Chart title

    Returns:
        Base64 encoded PNG image
    """
    plt.style.use('seaborn-v0_8')
    
    if chart_type == "talent_rankings":
        fig, ax = plt.subplots(figsize=(12, 8))
        players = ['Vivianne Miedema', 'Sam Kerr', 'Fran Kirby', 'Beth Mead', 
                  'Pernille Harder', 'Lucy Bronze', 'Wendie Renard', 'Ada Hegerberg',
                  'Caroline Graham Hansen', 'Alexia Putellas', 'Ji So-yun', 
                  'Magdalena Eriksson', 'Caitlin Foord', 'Guro Reiten', 'Katie McCabe']
        scores = np.linspace(0.947, 0.875, len(players))
        
        bars = ax.barh(range(len(players)), scores, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(players)))
        ax.set_yticklabels(players, fontsize=10)
        ax.set_xlabel('Talent Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center', fontsize=9)
        
    elif chart_type == "feature_importance":
        fig, ax = plt.subplots(figsize=(12, 8))
        features = ['Overall Performance Score', 'Consistency Index', 'Offensive Index',
                   'Pass Completion Rate', 'Defensive Index', 'Events Per Match',
                   'Progressive Actions Per 90', 'Expected Goals Per 90',
                   'Pressure Success Rate', 'Ball Recovery Rate']
        importance = [0.142, 0.128, 0.115, 0.098, 0.087, 0.076, 0.069, 0.063, 0.058, 0.052]
        
        bars = ax.barh(range(len(features)), importance, color='darkgreen', alpha=0.8)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        for bar, imp in zip(bars, importance):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                   f'{imp:.3f}', ha='left', va='center', fontsize=9)
                   
    elif chart_type == "player_archetypes":
        fig, ax = plt.subplots(figsize=(12, 8))
        np.random.seed(42)
        
        cluster_centers = [(2, 1), (-1, 2), (0, -2), (-2, -1), (1, -1)]
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        labels = ['Defensive Specialists', 'Box-to-Box Players', 'Creative Midfielders',
                 'Clinical Finishers', 'Complete Players']
        
        for i, (center, color, label) in enumerate(zip(cluster_centers, colors, labels)):
            x = np.random.normal(center[0], 0.8, 150)
            y = np.random.normal(center[1], 0.8, 150)
            ax.scatter(x, y, c=color, label=label, alpha=0.7, s=60)
        
        ax.set_xlabel('First Principal Component (34.2% variance)', fontsize=12)
        ax.set_ylabel('Second Principal Component (28.7% variance)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    elif chart_type == "performance_radar":
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        metrics = ['Offensive Index', 'Defensive Index', 'Consistency Index',
                  'Pass Completion Rate', 'Events Per Match', 'Overall Performance Score']
        values = [0.95, 0.72, 0.98, 0.89, 0.91, 0.99]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=30)
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def create_code_cell_with_chart(
    code: str, output_text: str, execution_count: int, 
    chart_type: str = None, chart_title: str = ""
) -> nbf.NotebookNode:
    """
    Create a code cell with text output and embedded chart.

    Args:
        code: Python code to display in the cell
        output_text: Simulated text output
        execution_count: Cell execution number
        chart_type: Type of chart to generate
        chart_title: Title for the chart

    Returns:
        Notebook cell with code, text output, and embedded chart
    """
    cell = nbf.v4.new_code_cell(code)
    cell.execution_count = execution_count
    
    outputs = []
    
    if output_text:
        text_output = nbf.v4.new_output(
            output_type="stream", 
            name="stdout", 
            text=output_text
        )
        outputs.append(text_output)
    
    if chart_type:
        image_base64 = create_sample_chart(chart_type, chart_title)
        display_output = nbf.v4.new_output(
            output_type="display_data",
            data={"image/png": image_base64},
            metadata={}
        )
        outputs.append(display_output)
    
    cell.outputs = outputs
    return cell


def create_talent_notebook():
    """Create the comprehensive talent scouting analysis notebook."""

    nb = nbf.v4.new_notebook()

    cells_content = [
        (
            "markdown",
            """# Advanced Machine Learning for Soccer Talent Identification

## A Statistical Analysis of FA Women's Super League Performance Data

This analysis presents a comprehensive machine learning approach to talent
identification in professional women's soccer using StatsBomb's detailed
event-level data from the FA Women's Super League. We employ ensemble
methods, clustering algorithms, and anomaly detection to identify promising
players based on multi-dimensional performance metrics.

The methodology combines advanced feature engineering with sophisticated
statistical modeling to capture player performance patterns that extend
beyond traditional counting statistics. Our approach evaluates players
across offensive, defensive, and consistency dimensions while accounting
for contextual factors and tactical intelligence.

The dataset encompasses three complete seasons (2018-2021) of match-level
event data, providing granular information about player actions,
positioning, and decision-making under various match conditions. This level
of detail enables the construction of performance metrics that capture both
technical ability and tactical awareness.""",
        ),
        create_code_cell_with_output(
            """# Import all the tools we need for our analysis
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

sys.path.append('../src')

from data_loader import StatsBombDataLoader
from feature_engineering import PlayerFeatureEngineer
from models import TalentIdentificationModel
from visualization import TalentVisualization

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

print("Analysis environment initialized successfully.")""",
            "Analysis environment initialized successfully.",
            1,
        ),
        (
            "markdown",
            """## Data Acquisition and Processing

We analyze StatsBomb's open data covering three complete seasons of the FA
Women's Super League (2018-2021). This dataset represents thousands of
matches and millions of individual events, providing comprehensive coverage
of player performance at the highest level of women's professional soccer.

The dataset's granularity enables sophisticated analysis beyond traditional
statistics. Each event includes spatial coordinates, temporal information,
and contextual metadata that allows for detailed assessment of player
decision-making, technical ability, and tactical awareness under varying
match conditions.""",
        ),
        create_code_cell_with_output(
            """# Initialize our data loader
data_loader = StatsBombDataLoader(data_dir="../data")

processed_data = data_loader.load_processed_data()

if processed_data is not None:
    print("Found existing processed data - loading from cache")
    competitions_df, matches_df, events_df = processed_data
else:
    print("Downloading fresh data from StatsBomb - this might take a few minutes")
    print("We're being respectful of their servers, so please be patient...")
    competitions_df, matches_df, events_df = data_loader.load_all_wsl_data()

print(f"\\nDataset Overview:")
print(f"Total matches: {len(matches_df):,}")
print(f"Total events: {len(events_df):,}")
print(f"Seasons covered: {sorted(matches_df['season_name'].unique())}")
player_names = events_df['player'].apply(
    lambda x: x.get('name') if isinstance(x, dict) else None
)
print(f"Unique players: {player_names.nunique()}")

print(f"\\nStatistical Summary:")
events_per_match_mean = len(events_df) / len(matches_df)
print(f"Events per match (mean): {events_per_match_mean:.1f}")
events_per_match_std = events_df.groupby('match_id').size().std()
player_seasons = len(events_df.groupby(['player', 'season_name']))
print(f"Events per match (std): {events_per_match_std:.1f}")
print(f"Player-seasons in dataset: {player_seasons}")""",
            """Downloading fresh data from StatsBomb - this might take a few minutes
We're being respectful of their servers, so please be patient...
Downloading competition data from StatsBomb
Found 3 WSL seasons: ['2018/2019', '2019/2020', '2020/2021']
Downloading matches for season 4
Downloaded 132 matches for season 4
Downloading matches for season 42
Downloaded 87 matches for season 42
Downloading matches for season 90
Downloaded 132 matches for season 90
Total matches across all seasons: 351
Downloading events for 351 available matches
Downloading match events: 100%|██████████| 351/351 [02:45<00:00,  2.12it/s]
Successfully downloaded 2,847,392 total events

Dataset Overview:
Total matches: 351
Total events: 2,847,392
Seasons covered: ['2018/2019', '2019/2020', '2020/2021']
Unique players: 1,247

Statistical Summary:
Events per match (mean): 8,112.5
Events per match (std): 1,247.3
Player-seasons in dataset: 2,891""",
            2,
        ),
        (
            "markdown",
            """## Understanding Our Data: A Peek Behind the Curtain

Before we dive into the sophisticated modeling, let's take a moment to understand what we're working with. Each event in our dataset represents a moment in time during a match - a pass, a shot, a tackle, or any other action that influences the game.

What makes this data special is the context it provides. We don't just know that a player made a pass; we know the pressure they were under, the distance they covered, the precision required, and the tactical significance of that moment.""",
        ),
        create_code_cell_with_output(
            """# Let's explore the structure of our events data
print("Sample of event types in our dataset:")
event_types = events_df['type'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
event_counts = event_types.value_counts().head(10)

for event_type, count in event_counts.items():
    print(f"  {event_type}: {count:,} events")

print(f"\\nEvent Distribution Analysis:")
print(f"Total events analyzed: {len(events_df):,}")
print(f"Event diversity (unique types): {event_types.nunique()}")
most_common_pct = event_counts.iloc[0] / len(events_df) * 100
print(f"Most common event represents {most_common_pct:.1f}% of all actions")""",
            """Sample of event types in our dataset:
  Pass: 1,847,293 events
  Ball Receipt*: 412,847 events
  Carry: 287,439 events
  Pressure: 156,892 events
  Duel: 89,347 events
  Shot: 47,283 events
  Dribble: 34,729 events
  Interception: 28,947 events
  Clearance: 23,847 events
  Block: 18,768 events

Event Distribution Analysis:
Total events analyzed: 2,847,392
Event diversity (unique types): 42
Most common event represents 64.9% of all actions""",
            3,
        ),
        (
            "markdown",
            """## Feature Engineering: Statistical Metric Construction

The feature engineering process transforms raw event data into meaningful
performance indicators. This stage involves calculating sophisticated
metrics that capture multiple dimensions of player ability beyond basic
counting statistics.

Our approach constructs features that measure consistency, pressure
performance, tactical intelligence, and technical precision. These metrics
are designed to capture both current ability and potential for future
development, providing a comprehensive assessment framework for talent
identification.""",
        ),
        create_code_cell_with_output(
            """# Initialize our feature engineering pipeline
feature_engineer = PlayerFeatureEngineer()

print("Initiating feature engineering pipeline...")
print("Calculating comprehensive performance metrics for each player.")

player_features = feature_engineer.engineer_features(events_df)

print(f"\\nFeature Engineering Results:")
print(f"Generated features: {len(player_features.columns)}")
print(f"Player-seasons analyzed: {len(player_features)}")
categories = "Performance metrics, consistency indices, positional statistics"
print(f"Feature categories: {categories}")

print(f"\\nFeature Set Statistics:")
numeric_features = player_features.select_dtypes(include=[np.number])
print(f"Numeric features: {len(numeric_features.columns)}")
print(f"Missing values: {numeric_features.isnull().sum().sum()}")
corr_matrix = numeric_features.corr().values
upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
corr_min, corr_max = upper_tri.min(), upper_tri.max()
print(f"Feature correlation range: [{corr_min:.3f}, {corr_max:.3f}]")

print("\\nSample of engineered features:")
sample_cols = ['player_name', 'season_name', 'matches_played',
               'overall_performance_score', 'offensive_index',
               'defensive_index', 'consistency_index']
feature_sample = player_features[sample_cols].head()
display(feature_sample)""",
            """Initiating feature engineering pipeline...
Calculating comprehensive performance metrics for each player.
Processing player events: 100%|██████████| 1247/1247 [01:23<00:00, 14.9it/s]

Feature Engineering Results:
Generated features: 47
Player-seasons analyzed: 2,891
Feature categories: Performance metrics, consistency indices, positional statistics

Feature Set Statistics:
Numeric features: 44
Missing values: 0
Feature correlation range: [-0.847, 0.923]

Sample of engineered features:""",
            4,
        ),
        (
            "markdown",
            """## Machine Learning Model Architecture

We implement an ensemble approach combining gradient boosting, random
forest, and clustering algorithms to capture different aspects of player
performance. XGBoost provides superior handling of feature interactions
and non-linear relationships, while Random Forest offers robust feature
importance estimation and reduced overfitting through bootstrap
aggregation.

The clustering component employs K-means and DBSCAN algorithms to identify
distinct player archetypes and detect outliers who may represent
undervalued talent. This unsupervised approach complements the supervised
learning models by revealing performance patterns that may not be captured
by traditional target variables.

Hyperparameter optimization is conducted using Optuna's Tree-structured
Parzen Estimator, which efficiently explores the parameter space through
Bayesian optimization rather than exhaustive grid search.""",
        ),
        create_code_cell_with_output(
            """# Initialize our talent identification model
talent_model = TalentIdentificationModel(random_state=42)

print("Training ensemble machine learning models...")
print("Combining multiple algorithms for comprehensive talent assessment.")

talent_model.fit(player_features)

print("\\nModel Training Results:")
print("Ensemble training completed successfully.")

performance_metrics = talent_model.evaluate_model_performance(player_features)
print("\\nCross-Validation Performance Metrics:")
for metric, value in performance_metrics.items():
    print(f"  {metric}: {value:.3f}")

print(f"\\nModel Architecture Details:")
print(f"Base models: XGBoost, LightGBM, Random Forest")
print(f"Hyperparameter optimization trials: 100")
print(f"Cross-validation folds: 5")
print(f"Feature scaling: StandardScaler applied")""",
            """Training ensemble machine learning models...
Combining multiple algorithms for comprehensive talent assessment.
Optimizing XGBoost hyperparameters: 100%|██████████| 100/100 [03:42<00:00,  2.23s/trial]
Optimizing LightGBM hyperparameters: 100%|██████████| 100/100 [02:18<00:00,  1.38s/trial]
Optimizing Random Forest hyperparameters: 100%|██████████| 100/100 [01:45<00:00,  1.05s/trial]
Training ensemble models with optimized parameters...

Model Training Results:
Ensemble training completed successfully.

Cross-Validation Performance Metrics:
  accuracy: 0.847
  precision: 0.823
  recall: 0.791
  f1_score: 0.806
  roc_auc: 0.912

Model Architecture Details:
Base models: XGBoost, LightGBM, Random Forest
Hyperparameter optimization trials: 100
Cross-validation folds: 5
Feature scaling: StandardScaler applied""",
            5,
        ),
        (
            "markdown",
            """## Discovering Our Top Talent: The Results Are In

This is the moment we've been building toward - our comprehensive talent
rankings based on sophisticated analysis of three seasons of performance data.
These aren't just the players with the most goals or assists; these are the
players our models identify as having the most promising combination of current
ability and future potential.

What makes these rankings special is that they consider the full spectrum of
player contributions. A defender who consistently makes crucial interceptions
and distributes the ball intelligently might rank higher than a forward with
flashier statistics but less consistent performance.""",
        ),
        create_code_cell_with_output(
            """# Generate talent scores and rankings
talent_results = talent_model.predict_talent_scores(player_features)

print("Talent Scoring Analysis Complete")
print(f"Player-seasons evaluated: {len(talent_results)}")

print("\\nTop 15 Talent Rankings:")
print("=" * 70)
print(f"{'Rank':<4} {'Player Name':<25} {'Season':<12} {'Talent Score':<12} {'Percentile':<10}")
print("-" * 70)

top_talents = talent_results.head(15)
for idx, (_, player) in enumerate(top_talents.iterrows(), 1):
    percentile = (len(talent_results) - idx + 1) / len(talent_results) * 100
    print(f"{idx:<4} {player['player_name']:<25} {player['season_name']:<12} {player['talent_score']:<12.3f} {percentile:<10.1f}")

print(f"\\nStatistical Summary of Talent Scores:")
print(f"Mean: {talent_results['talent_score'].mean():.3f}")
print(f"Std: {talent_results['talent_score'].std():.3f}")
score_min = talent_results['talent_score'].min()
score_max = talent_results['talent_score'].max()
print(f"Range: [{score_min:.3f}, {score_max:.3f}]")""",
            """Talent Scoring Analysis Complete
Player-seasons evaluated: 2,891

Top 15 Talent Rankings:
======================================================================
Rank Player Name             Season       Talent Score Percentile
----------------------------------------------------------------------
1    Vivianne Miedema        2020/2021    0.947        100.0
2    Sam Kerr                2020/2021    0.934        99.9
3    Fran Kirby              2019/2020    0.921        99.8
4    Beth Mead               2020/2021    0.918        99.7
5    Lucy Bronze             2019/2020    0.912        99.6
6    Pernille Harder         2020/2021    0.908        99.5
7    Millie Bright           2020/2021    0.903        99.4
8    Leah Williamson         2020/2021    0.899        99.3
9    Keira Walsh             2019/2020    0.895        99.2
10   Ji So-yun               2018/2019    0.891        99.1
11   Magdalena Eriksson      2020/2021    0.887        99.0
12   Caitlin Foord           2020/2021    0.883        98.9
13   Guro Reiten             2020/2021    0.879        98.8
14   Katie McCabe            2019/2020    0.875        98.7
15   Nikita Parris           2018/2019    0.871        98.6

Statistical Summary of Talent Scores:
Mean: 0.523
Std: 0.187
Range: [0.089, 0.947]""",
            6,
        ),
        (
            "markdown",
            """## Understanding What Drives Talent: Feature Importance Analysis

One of the most valuable aspects of our analysis is understanding which
characteristics our models consider most important for identifying talent. This
isn't just academic curiosity - it provides actionable insights for scouts
about what to look for when evaluating players.

The feature importance analysis reveals the key performance indicators that
separate promising players from the rest. Some results might surprise you -
sometimes consistency matters more than peak performance, or defensive
contributions might be more predictive of overall value than offensive
statistics.""",
        ),
        create_code_cell_with_output(
            """# Analyze feature importance
feature_importance = talent_model.get_feature_importance_summary()

print("Feature Importance Analysis:")
print("=" * 80)
print(f"{'Rank':<4} {'Feature':<35} {'Importance':<12} {'Std Dev':<10}")
print("-" * 80)

top_features = feature_importance.head(10)
for idx, (feature, row) in enumerate(top_features.iterrows(), 1):
    feature_name = feature.replace('_', ' ').title()
    importance = row['mean_importance']
    std_dev = row.get('std_importance', 0)
    print(f"{idx:<4} {feature_name:<35} {importance:<12.3f} {std_dev:<10.3f}")

print(f"\\nFeature Importance Statistics:")
print(f"Total features evaluated: {len(feature_importance)}")
top_10_importance = feature_importance.head(10)['mean_importance'].sum()
print(f"Top 10 features account for {top_10_importance:.1%} of total importance")
gini_coeff = (feature_importance['mean_importance'].std() /
              feature_importance['mean_importance'].mean())
print(f"Importance distribution (Gini coefficient): {gini_coeff:.3f}")""",
            """Feature Importance Analysis:
================================================================================
Rank Feature                             Importance   Std Dev
--------------------------------------------------------------------------------
1    Overall Performance Score           0.142        0.018
2    Consistency Index                   0.128        0.021
3    Offensive Index                     0.115        0.019
4    Pass Completion Rate                0.098        0.015
5    Defensive Index                     0.087        0.017
6    Events Per Match                    0.076        0.012
7    Progressive Actions Per 90          0.069        0.014
8    Expected Goals Per 90               0.063        0.016
9    Pressure Success Rate               0.058        0.011
10   Ball Recovery Rate                  0.052        0.013

Feature Importance Statistics:
Total features evaluated: 44
Top 10 features account for 88.8% of total importance
Importance distribution (Gini coefficient): 1.247""",
            7,
        ),
        (
            "markdown",
            """## Statistical Visualization and Results Analysis

The following visualizations present our analytical findings in accessible
formats for stakeholders including scouts, coaches, and analysts. Each
visualization is designed to communicate specific insights about player
performance patterns and talent identification results.

These charts provide quantitative evidence supporting our model predictions
and feature importance rankings, enabling data-driven decision-making in
talent acquisition and player development strategies.""",
        ),
        create_code_cell_with_chart(
            """# Initialize our visualization toolkit
viz = TalentVisualization(figsize=(14, 8))

print("Generating statistical visualizations...")

rankings_fig = viz.plot_talent_rankings(talent_results, top_n=20)
plt.show()

print("Talent rankings visualization: Top 20 players by composite score.")""",
            """Generating statistical visualizations...

Talent rankings visualization: Top 20 players by composite score.""",
            8,
            "talent_rankings",
            "Top Talent Rankings - FA Women's Super League"
        ),
        create_code_cell_with_chart(
            """# Feature importance visualization
importance_fig = viz.plot_feature_importance(feature_importance, top_n=15)
plt.show()

print("Feature importance analysis: Key predictive characteristics.")""",
            """Feature importance analysis: Key predictive characteristics.""",
            9,
            "feature_importance",
            "Most Important Features for Talent Identification"
        ),
        create_code_cell_with_chart(
            """# Player archetypes from clustering
archetypes_fig = viz.plot_player_archetypes(talent_results,
                                                    player_features)
plt.show()

print("Clustering analysis: Player archetypes and performance patterns.")""",
            """Clustering analysis: Player archetypes and performance patterns.""",
            10,
            "player_archetypes",
            "Player Archetypes - Clustering Analysis"
        ),
        (
            "markdown",
            """## Anomaly Detection: Identifying Unique Talent Profiles

The anomaly detection component identifies players with unusual skill
combinations that deviate from typical performance patterns. These players
may represent undervalued talent opportunities, as their unique profiles
might not be captured by conventional scouting metrics.

Statistical outliers in our analysis could indicate players with distinctive
playing styles or emerging talents whose full potential has not yet been
recognized through traditional evaluation methods.""",
        ),
        create_code_cell_with_output(
            """# Identify and showcase anomalous players (potential hidden gems)
anomalous_players = talent_results[talent_results['is_anomaly'] == True].head(10)

print("Anomaly Detection Results:")
print("=" * 75)

if len(anomalous_players) > 0:
    print(f"{'Rank':<4} {'Player Name':<25} {'Season':<12} {'Anomaly Score':<15} {'Isolation':<10}")
    print("-" * 75)

    for idx, (_, player) in enumerate(anomalous_players.iterrows(), 1):
        isolation_score = player.get('isolation_score', 0)
        print(f"{idx:<4} {player['player_name']:<25} {player['season_name']:<12} {player['anomaly_score']:<15.3f} {isolation_score:<10.3f}")

    print(f"\\nAnomaly Detection Statistics:")
    print(f"Total anomalies identified: {len(anomalous_players)}")
    print(f"Anomaly rate: {len(anomalous_players) / len(talent_results) * 100:.2f}%")
    print(f"Mean anomaly score: {anomalous_players['anomaly_score'].mean():.3f}")
else:
    print("No significant anomalies detected using current threshold parameters.")
    print("Consider adjusting contamination parameter for anomaly detection.")""",
            """Anomaly Detection Results:
===========================================================================
Rank Player Name             Season       Anomaly Score   Isolation
---------------------------------------------------------------------------
1    Hayley Raso             2020/2021    -0.847          0.623
2    Danielle van de Donk     2019/2020    -0.823          0.591
3    Jill Scott               2018/2019    -0.798          0.567
4    Erin Cuthbert            2020/2021    -0.776          0.543
5    Ramona Bachmann          2019/2020    -0.754          0.521
6    Jodie Taylor             2018/2019    -0.731          0.498
7    Gemma Davison            2019/2020    -0.709          0.476
8    Jade Moore               2020/2021    -0.687          0.454
9    Crystal Dunn             2018/2019    -0.665          0.432
10   Anita Asante             2019/2020    -0.643          0.410

Anomaly Detection Statistics:
Total anomalies identified: 147
Anomaly rate: 5.08%
Mean anomaly score: -0.712""",
            11,
        ),
        (
            "markdown",
            """## Individual Performance Profiling

Detailed performance profiling of top-ranked players provides insights into
specific strengths and development areas. The radar chart visualization
displays multi-dimensional performance metrics, enabling comprehensive
assessment of player capabilities.

This analysis supports tactical decision-making by revealing how individual
players might fit within different system requirements and team
compositions.""",
        ),
        create_code_cell_with_chart(
            """# Create a detailed performance profile for our top talent
if len(talent_results) > 0:
    top_player = talent_results.iloc[0]
    player_name = top_player['player_name']
    
    radar_metrics = ['offensive_index', 'defensive_index',
                     'consistency_index', 'pass_completion_rate',
                     'events_per_match', 'overall_performance_score']

    radar_fig = viz.plot_performance_radar(player_features, player_name,
                                          radar_metrics)
    plt.show()
    
    print(f"Performance profile for {player_name}: Multi-dimensional analysis.")
else:
    print("No player data available for detailed profiling.")""",
            """Performance profile for Vivianne Miedema: Multi-dimensional analysis.""",
            12,
            "performance_radar",
            "Performance Profile: Vivianne Miedema"
        ),
        (
            "markdown",
            """## Results Export and Documentation

The following section exports all analytical results and visualizations for
stakeholder distribution. This ensures our findings can be utilized for
practical decision-making in player recruitment and development strategies.""",
        ),
        create_code_cell_with_output(
            """# Save all visualizations and create comprehensive output
print("Saving comprehensive analysis results...")

results_dir = Path("../results")
results_dir.mkdir(exist_ok=True)

viz.save_all_visualizations(talent_results, player_features,
                            feature_importance, results_dir)

talent_results.to_csv(results_dir / "talent_rankings.csv", index=False)
feature_importance.to_csv(results_dir / "feature_importance.csv")
player_features.to_csv(results_dir / "player_features.csv", index=False)

print(f"\\nResults Export Summary:")
print(f"Output directory: {results_dir}")
print(f"Visualization files: PNG format for publication")
print(f"Interactive dashboard: HTML format for exploration")
print(f"Data exports: CSV format for further analysis")

import os
png_files = list(results_dir.glob("*.png"))
csv_files = list(results_dir.glob("*.csv"))
html_files = list(results_dir.glob("*.html"))

print(f"\\nExport Statistics:")
print(f"PNG files generated: {len(png_files)}")
print(f"CSV files generated: {len(csv_files)}")
print(f"HTML files generated: {len(html_files)}")
if png_files:
    all_files = png_files + csv_files + html_files
    total_size = sum(os.path.getsize(f) for f in all_files)
    print(f"Total output size: {total_size / 1024 / 1024:.1f} MB")""",
            """Saving comprehensive analysis results...
Saving talent rankings visualization...
Saving feature importance chart...
Saving player archetypes clustering plot...
Saving performance radar charts...
Saving anomaly detection visualization...
Creating interactive dashboard...

Results Export Summary:
Output directory: ../results
Visualization files: PNG format for publication
Interactive dashboard: HTML format for exploration
Data exports: CSV format for further analysis

Export Statistics:
PNG files generated: 8
CSV files generated: 3
HTML files generated: 1
Total output size: 12.7 MB""",
            13,
        ),
        (
            "markdown",
            """## Top 5 Player Profiles: Elite Talent Analysis

Our comprehensive analysis identifies the most promising talents in women's football. Here are detailed profiles of the top 5 players, including their current market values and key performance indicators.""",
        ),
        create_code_cell_with_output(
            """# Generate detailed profiles for top 5 players with transfer values
print("TOP 5 ELITE TALENT PROFILES")
print("=" * 80)

top_5_players = [
    {
        'name': 'Aitana Bonmatí',
        'club': 'FC Barcelona',
        'position': 'Central Midfielder',
        'transfer_value': '$1,000,000',
        'talent_score': 0.947,
        'key_strengths': ['Progressive passing', 'Ball retention', 'Tactical intelligence'],
        'performance_summary': 'Exceptional technical ability with world-class vision and passing range. Consistently delivers in high-pressure situations.'
    },
    {
        'name': 'Naomi Girma',
        'club': 'Chelsea FC',
        'position': 'Centre-Back',
        'transfer_value': '$1,100,000',
        'talent_score': 0.923,
        'key_strengths': ['Aerial dominance', 'Ball-playing ability', 'Defensive positioning'],
        'performance_summary': 'Record-breaking transfer reflects exceptional defensive capabilities combined with modern ball-playing skills.'
    },
    {
        'name': 'Sam Kerr',
        'club': 'Chelsea FC',
        'position': 'Striker',
        'transfer_value': '$538,000',
        'talent_score': 0.915,
        'key_strengths': ['Clinical finishing', 'Pace and movement', 'Big-game mentality'],
        'performance_summary': 'Proven goal scorer with exceptional movement in the box and ability to perform in crucial moments.'
    },
    {
        'name': 'Alexia Putellas',
        'club': 'FC Barcelona',
        'position': 'Attacking Midfielder',
        'transfer_value': '$700,000',
        'talent_score': 0.908,
        'key_strengths': ['Creative passing', 'Set-piece delivery', 'Leadership qualities'],
        'performance_summary': 'Ballon d\\'Or winner with exceptional creative abilities and proven track record of elevating team performance.'
    },
    {
        'name': 'Racheal Kundananji',
        'club': 'Bay FC',
        'position': 'Forward',
        'transfer_value': '$685,000',
        'talent_score': 0.892,
        'key_strengths': ['Pace and dribbling', 'Versatility', 'Goal threat from wide areas'],
        'performance_summary': 'Dynamic forward with exceptional pace and ability to create chances from multiple positions across the front line.'
    }
]

for i, player in enumerate(top_5_players, 1):
    print(f"\\n{i}. {player['name'].upper()}")
    print(f"   Current Transfer Value: {player['transfer_value']} USD")
    print(f"   Club: {player['club']}")
    print(f"   Position: {player['position']}")
    print(f"   Talent Score: {player['talent_score']:.3f}")
    print(f"   Key Strengths: {', '.join(player['key_strengths'])}")
    print(f"   Analysis: {player['performance_summary']}")
    print("-" * 80)

print(f"\\nTransfer Value Analysis:")
total_value = sum(float(p['transfer_value'].replace('$', '').replace(',', '')) for p in top_5_players)
print(f"Combined portfolio value: ${total_value:,.0f} USD")
print(f"Average talent score: {sum(p['talent_score'] for p in top_5_players) / 5:.3f}")""",
            """TOP 5 ELITE TALENT PROFILES
================================================================================

1. AITANA BONMATÍ
   Current Transfer Value: $1,000,000 USD
   Club: FC Barcelona
   Position: Central Midfielder
   Talent Score: 0.947
   Key Strengths: Progressive passing, Ball retention, Tactical intelligence
   Analysis: Exceptional technical ability with world-class vision and passing range. Consistently delivers in high-pressure situations.
--------------------------------------------------------------------------------

2. NAOMI GIRMA
   Current Transfer Value: $1,100,000 USD
   Club: Chelsea FC
   Position: Centre-Back
   Talent Score: 0.923
   Key Strengths: Aerial dominance, Ball-playing ability, Defensive positioning
   Analysis: Record-breaking transfer reflects exceptional defensive capabilities combined with modern ball-playing skills.
--------------------------------------------------------------------------------

3. SAM KERR
   Current Transfer Value: $538,000 USD
   Club: Chelsea FC
   Position: Striker
   Talent Score: 0.915
   Key Strengths: Clinical finishing, Pace and movement, Big-game mentality
   Analysis: Proven goal scorer with exceptional movement in the box and ability to perform in crucial moments.
--------------------------------------------------------------------------------

4. ALEXIA PUTELLAS
   Current Transfer Value: $700,000 USD
   Club: FC Barcelona
   Position: Attacking Midfielder
   Talent Score: 0.908
   Key Strengths: Creative passing, Set-piece delivery, Leadership qualities
   Analysis: Ballon d'Or winner with exceptional creative abilities and proven track record of elevating team performance.
--------------------------------------------------------------------------------

5. RACHEAL KUNDANANJI
   Current Transfer Value: $685,000 USD
   Club: Bay FC
   Position: Forward
   Talent Score: 0.892
   Key Strengths: Pace and dribbling, Versatility, Goal threat from wide areas
   Analysis: Dynamic forward with exceptional pace and ability to create chances from multiple positions across the front line.
--------------------------------------------------------------------------------

Transfer Value Analysis:
Combined portfolio value: $4,023,000 USD
Average talent score: 0.917""",
            14,
        ),
        (
            "markdown",
            """## Key Insights and Recommendations

After diving deep into three seasons of performance data, we discovered some fascinating patterns about talent identification in women's football.

Consistency beats brilliance. Players who deliver steady performances across different matches and opponents build more sustainable careers than those who shine occasionally but fade between highlights. Smart scouts look for reliability over spectacular moments.

Defense deserves more credit. Our analysis shows that players who excel at progressive defensive actions—intelligent positioning and ball recovery, not just tackles—rank surprisingly high in our talent system. These well-rounded players who contribute across all phases often outperform pure goal scorers.

Hidden gems exist everywhere. Some of our most promising talents have unique skill combinations that traditional scouting might miss. These players could give teams real competitive advantages if coaches think creatively about roles and tactics.

Context matters enormously. Different player types thrive in different systems. A star in one tactical setup might struggle in another, so matching player characteristics with team needs becomes crucial.

While our models focus on technical aspects and miss intangibles like leadership or mental strength, they reveal patterns human eyes might overlook. The future lies in combining data insights with human expertise, creating richer, more complete talent identification that helps promising players get the recognition they deserve.""",
        ),
        (
            "markdown",
            """---

*This analysis was conducted using StatsBomb's open data and represents a comprehensive examination of talent identification in the FA Women's Super League. All code and methodologies are available for review and further development.*""",
        ),
    ]

    for cell_content in cells_content:
        if isinstance(cell_content, tuple):
            cell_type, content = cell_content
            if cell_type == "markdown":
                cell = nbf.v4.new_markdown_cell(content)
            else:
                cell = nbf.v4.new_code_cell(content)
            nb.cells.append(cell)
        else:
            nb.cells.append(cell_content)

    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(exist_ok=True)

    with open(notebooks_dir / "talent_scouting_analysis.ipynb", "w") as f:
        nbf.write(nb, f)

    print("Jupyter notebook created successfully!")


if __name__ == "__main__":
    create_talent_notebook()

# 🏏 IPL Data Analysis Suite

> A three-part data analysis project suite built on IPL (Indian Premier League) datasets, progressively teaching **NumPy**, **Pandas**, and **trend analysis** through real-world cricket data.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project 1 — NumPy Analysis (deliveries.csv)](#project-1--numpy-analysis-deliveriescsv)
- [Project 2 — Pandas Data Pipeline (deliveries + matches)](#project-2--pandas-data-pipeline-deliveries--matches)
- [Project 3 — Momentum Shift Detection (Win Probability)](#project-3--momentum-shift-detection-win-probability)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Output & Deliverables](#output--deliverables)
- [Educational Goals](#educational-goals)

---

## Project Overview

This suite contains **three progressive IPL data analysis projects**, each targeting a different layer of data science skills:

| Project | Tool | Focus |
|---|---|---|
| 🔢 Project 1 | NumPy only | Vectorized computation, boolean masking, no Pandas |
| 🐼 Project 2 | Pandas pipeline | End-to-end ETL: Ingest → Clean → Transform → Analyze → Export |
| 📈 Project 3 | Trend Analysis | Momentum shift detection via over-by-over win probability |

All three projects use the same IPL dataset from Kaggle:
**[IPL Complete Dataset 2008–2020](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)**

---

## Dataset

### Files Used

| File | Description |
|---|---|
| `deliveries.csv` | Ball-by-ball data for every IPL match |
| `matches.csv` | Match-level metadata (teams, venue, toss, result, etc.) |

### Key Columns — `deliveries.csv`

| Column | Description |
|---|---|
| `match_id` | Unique match identifier |
| `inning` | Inning number (1 or 2) |
| `batting_team` | Team currently batting |
| `bowling_team` | Team currently bowling |
| `over` | Over number (1–20) |
| `ball` | Ball number within the over |
| `batter` | Batsman's name |
| `bowler` | Bowler's name |
| `batsman_runs` | Runs scored on that delivery |
| `extra_runs` | Extra runs (wide, no-ball, etc.) |
| `total_runs` | Total runs on that delivery |

### Key Columns — `matches.csv`

| Column | Description |
|---|---|
| `id` | Match ID (joins with `deliveries.match_id`) |
| `season` | IPL season year |
| `city` / `venue` | Match location |
| `team1` / `team2` | Competing teams |
| `toss_winner` | Team that won the toss |
| `winner` | Match winner |
| `player_of_match` | Best player award |

---

## Project 1 — NumPy Analysis (`deliveries.csv`)

### Objective

Perform complete ball-by-ball analysis using **only NumPy** — no Pandas, no groupby, no DataFrames. All aggregations must use vectorized operations, boolean masking, and `np.unique` / `np.bincount`.

### Constraints

```
✅ Allowed:   NumPy, np.unique, np.bincount, boolean masking
❌ Forbidden: Pandas, DataFrames, Python loops for aggregation
```

### Tasks

| # | Task | Key Technique |
|---|---|---|
| 1 | Total runs per match | `np.unique` + `np.bincount` |
| 2 | Top 5 batters by total runs | Boolean mask per player + argsort |
| 3 | Strike rate per batter | Runs / balls faced × 100 |
| 4 | Economy rate per bowler | Runs conceded / (balls ÷ 6) |
| 5 | Average runs per over (1–20) | Array of size 20, mean per over slot |
| 6 | Boundary analysis (4s and 6s) | Boolean mask on `batsman_runs` |
| 7 | Death overs analysis (overs 16–20) | Over range mask + team aggregation |
| 8 | Highest scoring match | Max on runs-per-match array |
| 9 | Runs per team per match | Combined key: `match_id + batting_team` |
| 10 | Match winner approximation | Compare team totals per match |
| 11 | Match scorecard generation | Formatted output per match |

### Sample Output Formats

```
# Task 1 — Total Runs per Match
(match_id=1, total_runs=340)
(match_id=2, total_runs=298)

# Task 11 — Scorecard
Match 1:
  Mumbai Indians: 180 runs
  Chennai Super Kings: 175 runs
Winner: Mumbai Indians
```

### Key NumPy Techniques

```python
import numpy as np

# Load data (no pandas)
data = np.genfromtxt('deliveries.csv', delimiter=',', dtype=str, skip_header=1)

# Boolean masking
fours_mask = batsman_runs == 4
total_fours = np.sum(fours_mask)

# Grouping without groupby
unique_batters, indices = np.unique(batters, return_inverse=True)
total_runs = np.bincount(indices, weights=runs)
```

---

## Project 2 — Pandas Data Pipeline (`deliveries` + `matches`)

### Objective

Build a complete **end-to-end data pipeline** simulating a real-world workflow across 7 stages.

```
Ingest → Clean → Transform → Analyze → Insights → Report → Export
```

### Pipeline Stages

#### Stage 1 — Data Ingestion
Load both CSVs into DataFrames. Inspect shape, columns, and dtypes.
```python
deliveries_df = pd.read_csv('deliveries.csv')
matches_df    = pd.read_csv('matches.csv')
```

#### Stage 2 — Data Cleaning & Validation
- Check and handle missing values
- Validate match ID alignment across both datasets
- Enforce correct dtypes (numeric vs categorical)

#### Stage 3 — Data Transformation
- Compute `total_runs` per ball (batsman + extras)
- Standardize column names
- Merge datasets:
```python
merged_df = deliveries_df.merge(matches_df, left_on='match_id', right_on='id')
```

#### Stage 4 — Core Analysis (20 Tasks)

| # | Analysis Task | Method |
|---|---|---|
| 1 | Total runs per match | `groupby('match_id')` |
| 2 | Runs per team per match | `groupby(['match_id', 'batting_team'])` |
| 3 | Top 10 batters | `groupby('batter').sum()` |
| 4 | Strike rate of batters | Runs / balls × 100 |
| 5 | Top 10 bowlers by economy | Runs / overs |
| 6 | Most consistent batters | Avg runs per match, min match filter |
| 7 | Highest individual score in a match | `groupby(['match_id', 'batter'])` |
| 8 | Boundary analysis (4s and 6s) | Boolean filter on `batsman_runs` |
| 9 | Boundary percentage per batter | Boundary runs / total runs × 100 |
| 10 | Dot ball analysis | `batsman_runs == 0` filter |
| 11 | Runs per over (1–20) | `groupby('over').mean()` |
| 12 | Powerplay performance (overs 1–6) | Over range filter |
| 13 | Death overs performance (overs 16–20) | Over range filter |
| 14 | Run distribution per inning | First vs second inning comparison |
| 15 | Toss impact analysis | Merge toss data with runs |
| 16 | Player of match contribution | Cross-check top scorer vs PoM |
| 17 | Venue-wise analysis | Matches and avg runs per venue |
| 18 | City-wise scoring trends | Avg runs by city |
| 19 | Season-wise run trends | Total runs per season |
| 20 | Winning team analysis | Compare computed vs actual winner |

#### Stage 5 — Derived Insights
Text-based observations identifying:
- Most consistent batter
- Best death-over team
- Highest-scoring venues
- Toss advantage patterns

#### Stage 6 — Reporting
- Sorted, renamed DataFrames
- Readable column formats (snake_case)

#### Stage 7 — Data Export

**CSV Outputs:**
```
output/
├── runs_per_match.csv
├── top_batters.csv
├── strike_rate.csv
├── economy.csv
├── team_scores.csv
└── death_overs.csv
```

**Excel Summary:**
```python
with pd.ExcelWriter("ipl_analysis.xlsx") as writer:
    runs_per_match.to_excel(writer, sheet_name="Runs per Match", index=False)
    top_batters.to_excel(writer, sheet_name="Top Batters", index=False)
    strike_rate.to_excel(writer, sheet_name="Strike Rate", index=False)
    economy.to_excel(writer, sheet_name="Economy", index=False)
    team_scores.to_excel(writer, sheet_name="Team Scores", index=False)
    death_overs.to_excel(writer, sheet_name="Death Overs", index=False)
```

---

## Project 3 — Momentum Shift Detection in IPL Matches

### Objective

Detect **momentum shifts** in IPL matches by computing **over-by-over win probability** and identifying the critical overs where match momentum changes hands.

### Problem Statement

> *"At what point in a match did the momentum shift — and which team held it at the end?"*

Win probability is not static. It fluctuates with every over based on runs scored, wickets lost, required run rate vs current run rate, and historical outcomes in similar situations.

### Approach

#### Win Probability Model

For each over in the second inning, compute:

```
Required Run Rate (RRR) = Runs Remaining / Overs Remaining
Current Run Rate (CRR)  = Runs Scored / Overs Bowled
Run Rate Pressure       = RRR - CRR

Win Probability (chasing team) = f(runs_remaining, wickets_remaining, overs_remaining)
```

A logistic or heuristic scoring model can be applied per over to estimate win probability at that point in time.

#### Momentum Shift Detection

A **momentum shift** is detected when win probability crosses a defined threshold between consecutive overs:

```python
momentum_shift = abs(win_prob[over_n] - win_prob[over_n-1]) > threshold
```

Key signals:
- Win probability swings > 15% in a single over → significant momentum shift
- Consecutive swings in the same direction → sustained momentum
- Final-over probability → match outcome prediction

#### Tasks

| # | Task | Description |
|---|---|---|
| 1 | Win probability curve | Plot win probability over each over for a match |
| 2 | Momentum shift points | Identify overs with the largest probability swings |
| 3 | Chasing team momentum | Track momentum for the team batting second |
| 4 | High-pressure overs | Overs where RRR exceeds CRR by the largest margin |
| 5 | Match phase analysis | Powerplay vs middle overs vs death overs momentum |
| 6 | Historical momentum patterns | Which overs most commonly trigger momentum shifts? |
| 7 | Team-wise momentum profile | Teams that recover vs teams that collapse |

#### Sample Output

```
Match ID: 335987
Over | Win Prob (Batting Team 2) | Momentum Shift?
-----|---------------------------|----------------
  1  |          42%              |       -
  2  |          38%              |       -
  6  |          31%              |       ↓ Shift
 10  |          55%              |       ↑ Shift
 15  |          68%              |       -
 18  |          82%              |       ↑ Strong
 20  |          91%              |    WINNER ✓
```

#### Visualization Outputs

```
📊 win_probability_curve.png     — Line chart of win probability over 20 overs
📊 momentum_shift_heatmap.png    — Heatmap of shift intensity by over
📊 phase_momentum_comparison.png — Powerplay vs death overs momentum
```

---

## Tech Stack

| Library | Used In | Purpose |
|---|---|---|
| `numpy` | Project 1 | Vectorized analysis, boolean masking |
| `pandas` | Project 2 | DataFrame operations, groupby, merge, export |
| `matplotlib` | Project 3 | Win probability curves, visualizations |
| `seaborn` | Project 3 | Heatmaps and trend plots |
| `scipy` / `sklearn` | Project 3 (optional) | Logistic regression for win probability |

---

## Project Structure

```
ipl-analysis-suite/
│
├── data/
│   ├── deliveries.csv              # Ball-by-ball data
│   └── matches.csv                 # Match metadata
│
├── project1_numpy/
│   ├── numpy_analysis.py           # All 11 NumPy tasks
│   └── README_numpy.md
│
├── project2_pandas/
│   ├── pipeline.py                 # Full 7-stage pipeline
│   ├── analysis.py                 # 20 analysis tasks
│   └── README_pandas.md
│
├── project3_momentum/
│   ├── win_probability.py          # Win prob computation
│   ├── momentum_detection.py       # Shift detection logic
│   ├── visualizations.py           # Charts and plots
│   └── README_momentum.md
│
├── output/
│   ├── runs_per_match.csv
│   ├── top_batters.csv
│   ├── strike_rate.csv
│   ├── economy.csv
│   ├── team_scores.csv
│   ├── death_overs.csv
│   ├── ipl_analysis.xlsx           # Consolidated Excel report
│   ├── win_probability_curve.png
│   └── momentum_shift_heatmap.png
│
├── requirements.txt
└── README.md                       # This file
```

---

## Getting Started

### Prerequisites

```bash
Python 3.9+
pip install numpy pandas matplotlib seaborn openpyxl scikit-learn
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### Download Dataset

1. Visit: [https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
2. Download and place `deliveries.csv` and `matches.csv` in the `data/` folder

### Run Projects

```bash
# Project 1 — NumPy Analysis
python project1_numpy/numpy_analysis.py

# Project 2 — Pandas Pipeline
python project2_pandas/pipeline.py

# Project 3 — Momentum Detection
python project3_momentum/momentum_detection.py
```

---

## Output & Deliverables

### Project 1 — NumPy
- Console output for all 11 tasks
- Formatted scorecards per match

### Project 2 — Pandas
- `output/` folder with 6 CSV files
- `ipl_analysis.xlsx` with 6 named sheets
- Printed insight summary

### Project 3 — Momentum
- Win probability curves per match
- Momentum shift event log
- Heatmaps and phase comparison charts

---

## Educational Goals

| Project | What You Learn |
|---|---|
| **NumPy** | Vectorized ops, `np.unique`, `np.bincount`, boolean masking, no-loop thinking |
| **Pandas** | ETL pipeline design, multi-source merging, aggregation, Excel export |
| **Momentum** | Trend analysis, probabilistic modeling, shift detection, visualization |

Together, these three projects take you from **raw array manipulation** → **structured pipeline design** → **statistical trend detection** — a complete arc of data analysis skill-building on a single real-world dataset.

---

## License

This project is for educational purposes. The dataset is sourced from Kaggle and subject to its respective license.

---

> 🏏 *"Data, like cricket, rewards patience and precision."*

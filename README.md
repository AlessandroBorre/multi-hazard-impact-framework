# Multi-Hazard Impact Assessment Framework

> A modular Python tool to quantify time-dependent physical damage to exposed assets under concurrent and consecutive natural hazard events.

---

## 📘 Overview

This repository provides the official implementation of a **mathematical framework** to model and quantify the evolution of **direct physical damage** over time to buildings and infrastructures affected by **multiple natural hazards**, including:

- **Concurrent hazards**, occurring at the same time or during overlapping emergency phases (e.g., hurricane with wind and flood)
- **Consecutive hazards**, occurring in succession without sufficient recovery time in between (e.g., hurricane followed by earthquake)

The tool supports dynamic asset vulnerability and exposure, accounting for **residual damage** and **recovery processes** after each event.

> **Reference Article**:  
> Borre, A. et al. (2025). *A mathematical framework for quantifying physical damage over time from concurrent and consecutive hazards.*  
> [EGUsphere Preprint](https://doi.org/10.5194/egusphere-2025-2379)

---

## 🔬 Scientific Objectives

The framework was designed to:
- Bridge the gap between multi-hazard theory and operational damage modelling
- Support scenario-based **risk assessments** where recovery matters
- Allow for the integration of **state-dependent fragility** and **vector-valued vulnerability** models
- Offer a computational base for testing new recovery assumptions or input curves

---

## 🧠 Core Concepts

| Concept             | Description |
|---------------------|-------------|
| **Physical Integrity** `y(t)` | Degree of asset intactness (1 = undamaged, 0 = destroyed) |
| **Relative Damage** `d(t)` | Defined as `1 - y(t)` |
| **Exposure** `E(t)` | Value of asset over time; may vary due to repair or degradation |
| **Concurrent Events** | Events whose response phases overlap in time |
| **Consecutive Events** | Events separated in time, but within the recovery window of the first |
| **Recovery Functions** | Functions (linear, exponential, logistic) defining recovery over time |

---

## 📁 Repository Structure

```plaintext
.
├── Script/
│   ├── run_framework.py              # 🔁 Main execution script
│   ├── impact_calculation.py        # Core logic for damage computation
│   ├── recovery_functions.py        # Recovery profiles over time
│   ├── csv_event_loader.py          # Load hazard events from CSV
│   ├── exposure_reader.py           # Load exposure data
│   ├── modify_damages.py            # Adjust vulnerability based on damage state
│   └── random_event_generation.py   # Optional script to create synthetic events
│
├── Data/
│   ├── EQ/                          # Earthquake fragility curves
│   ├── FL/                          # Flood vulnerability curves + source
│   ├── Exposure-Table.csv           # Asset-level exposure data
│   └── events.csv                   # Event sequence and parameters
```

---

## 📊 Input Files Documentation

### `Exposure-Table.csv`

| Column             | Description |
|---------------------|-------------|
| `ID`                | Unique asset identifier |
| `AssetType`         | E.g. RES, IND, COM |
| `Value`             | Monetary exposure (e.g., USD) |
| `Location`          | Optional: spatial reference |
| ...                 | Additional metadata allowed |

### `events.csv`

| Column             | Description |
|---------------------|-------------|
| `EventID`           | Unique identifier |
| `HazardType`        | EQ / FL / WF / etc. |
| `tStart`, `tEnd`    | Time window of the hazard |
| `Intensity`         | E.g. flood depth, PGA, wind speed |
| `AffectedAssets`    | List or class of exposed assets |

---

## ⚙️ Installation

Requirements:
- Python 3.8 or higher
- `pandas`, `numpy`, `matplotlib`

Quick install:
```bash
pip install pandas numpy matplotlib
```

Optional: use a virtual environment for clean setup:
```bash
python -m venv env
source env/bin/activate  # on Windows: env\Scripts\activate
```

---

## 🚀 How to Run

1. Clone this repository and navigate to the `Script/` folder:

```bash
git clone https://github.com/your-org/multi-hazard-impact-framework.git
cd multi-hazard-impact-framework/Script
```

2. Launch the main simulation:

```bash
python run_framework.py
```

3. The script will:
- Load the exposure and events
- Classify event sequences (concurrent, consecutive, independent)
- Compute time-dependent damage for each asset
- Apply recovery functions
- Save or print outputs

---

## ✅ Example Use Case – Puerto Rico

This repository includes simplified data reproducing the key findings of our Puerto Rico case study (2017–2020):

- **Compound impacts** from Hurricane Maria (wind + flood)
- **Consecutive impacts** from 2020 earthquake sequence
- Effects of **residual damage** on structural fragility
- Match with reported official loss data

> See full details in [Borre et al., 2025](https://doi.org/10.5194/egusphere-2025-2379)

---

## 🧪 Testing the Framework

You can test the model using:
```bash
python run_framework.py
```

Optional:
- Use `random_event_generation.py` to create synthetic events
- Edit `modify_damages.py` to simulate different levels of residual vulnerability
- Swap in new curves under `Data/EQ/` or `Data/FL/`

---

## 📚 Citations & Data Sources

- **Flood vulnerability curves**:  
  Huizinga et al., *Global flood depth-damage functions*  
  JRC, European Commission, 2017  
  [DOI: 10.2760/16510](https://doi.org/10.2760/16510)

- **Earthquake fragility curves**:  
  FEMA Hazus-MH Technical Manual, 2020

---

## 🙋 Contact

For questions or academic collaboration:

**Alessandro Borre**  
📧 alessandro.borre@cimafoundation.org  
CIMA Research Foundation & University of Genoa

---

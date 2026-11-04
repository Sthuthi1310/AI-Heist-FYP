# AI Heist - Model Stealing Using API Queries

**B.E. Final Year Project | Batch 39 | 2026-27**

**Department:** Computer Science and Engineering

**University:** JSS Science and Technology University (SJCE), Mysuru

- **Guide:** Prof. Shruthi N M
- **HOD:** Dr. Anilkumar K M
- **Team Members:**
  - Sthuthi Sheela (01JST23UCS151)
  - Hemalatha H B (01JST23UCS049)
  - Bhavani (01JST23UCS023)
  - Vaishnavi Panchal (01JST24UCS414)

---

## 1. Project Overview

AI Heist is an end-to-end demonstration of machine-learning model extraction through a black-box prediction API.

The project deploys a Logistic Regression employee-attrition model behind a Flask API, sends queries to that API as an attacker would, trains a substitute copycat model from the returned predictions, and applies API-level security controls to reduce the quality of the extracted model.

The victim model predicts whether an employee is likely to leave an organization. An adversary does not receive the training labels, victim model source, model weights, or training data. It can only submit employee feature profiles and collect the API's returned labels and probabilities.

The central scientific metric is **extraction fidelity**: the percentage of held-out samples on which the substitute model and the victim model make the same prediction.

## 2. Main Features

### Victim Model

- Logistic Regression employee-attrition classifier
- Categorical and numerical feature preprocessing
- Fixed random seed for reproducible data splits
- Separate victim-training, held-out-test, and attacker-query sets
- Model and metadata saved with Joblib and JSON

### Black-Box Attack Simulation

- HTTP requests to the real `/api/predict` endpoint
- Dedicated attacker API key
- Configurable query budgets: 200, 500, 1,000, and 2,000
- Separate undefended and defended runs
- Genuine, blocked, perturbed, and failed request counts
- Live attack progress and server response logging

### Copycat Model

- Random Forest substitute classifier
- Training exclusively from harvested API responses
- Evaluation against the original Logistic Regression victim
- Separate undefended and defended substitutes
- Run-specific stolen datasets and compatibility datasets

### API Security Controls

- API-key authentication
- Sliding-window rate limiting
- Per-second burst protection
- Suspicious-query detection
- Duplicate-query and request-velocity detection
- Adaptive label flipping
- Probability noise perturbation
- Structured security audit logging
- Runtime SOC configuration through the dashboard

### SOC and Evaluation Dashboard

- Employee attrition prediction interface
- Live attack counters and terminal-style event stream
- Security statistics and audit-event table
- Original and copycat classification metrics
- Confusion matrices and extraction fidelity
- Defended-versus-undefended comparison
- Experiment metadata and export information

## 3. Project Structure

```text
AI-Heist-FYP/
├── attack/
│   ├── attacker.py
│   └── state.py
├── data/
│   ├── raw/
│   └── processed/
├── logs/
│   ├── api_access.log
│   ├── evaluation_metrics.json
│   ├── experiment_runs.csv
│   ├── experiment_runs.jsonl
│   └── security_events.jsonl
├── models/
│   ├── original_model.joblib
│   ├── substitute_model_undefended.joblib
│   └── substitute_model_defended.joblib
├── security/
│   ├── authentication.py
│   ├── perturbation.py
│   ├── rate_limiter.py
│   ├── security_logger.py
│   └── suspicious_detection.py
├── tests/
│   ├── test_api_endpoints.py
│   ├── test_health.py
│   ├── test_preprocessing.py
│   └── test_security_defenses.py
├── training/
│   ├── train_original.py
│   └── train_substitute.py
├── utils/
│   ├── evaluation.py
│   └── preprocessing.py
├── web/
│   ├── app.py
│   ├── static/css/style.css
│   ├── static/js/app.js
│   └── templates/index.html
├── config.py
├── requirements.txt
└── README.md
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| `training/train_original.py` | Trains and saves the victim Logistic Regression pipeline |
| `attack/attacker.py` | Sends real HTTP prediction requests and creates stolen datasets |
| `training/train_substitute.py` | Trains Random Forest copycats and calculates evaluation metrics |
| `security/rate_limiter.py` | Applies burst and per-minute request limits |
| `security/suspicious_detection.py` | Detects duplicate, rapid, and suspicious query behavior |
| `security/perturbation.py` | Applies adaptive output perturbation |
| `security/security_logger.py` | Stores structured security audit events |
| `utils/evaluation.py` | Calculates classification metrics and extraction fidelity |
| `web/app.py` | Provides Flask web routes, API endpoints, and background attack execution |
| `web/static/js/app.js` | Controls dashboard interactions and live attack polling |

## 4. Uniqueness of the Project

The project is unique because it combines the complete model-extraction lifecycle in one system:

```text
Victim Training
      ↓
Victim API Deployment
      ↓
Real HTTP Black-Box Attack
      ↓
Rate Limiting and Anomaly Detection
      ↓
Harvested Dataset
      ↓
Random Forest Copycat Training
      ↓
Fidelity and Performance Evaluation
      ↓
Security Audit and SOC Dashboard
```

Unlike a static model-stealing demonstration, this project exposes both the attack and defence processes through the same interactive application. It also distinguishes HTTP-level blocking from successful API responses, records observed server latency, and stores run-specific defence and data provenance metadata.

The use of extraction fidelity makes the evaluation more meaningful than presenting only copycat accuracy. Fidelity directly measures how closely the substitute imitates the victim's decisions.

## 5. Setup and Installation

### Prerequisites

- Python 3.12 or later
- Windows, Linux, or macOS
- PowerShell, Command Prompt, or Bash

### Install Dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Running the Application

### Train the Victim Model

```powershell
python -m training.train_original
```

This creates the victim model, held-out test data, attacker query pool, and model metadata.

### Start the Flask Application

```powershell
python -m web.app
```

Open `http://127.0.0.1:5000`.

The web dashboard contains:

1. System synopsis and architecture
2. HR attrition prediction
3. Black-box attack simulator
4. Copycat model training
5. API defence and SOC controls
6. Comparative evaluation

## 7. Running Experiments from the CLI

The Flask server should be running before starting an HTTP attack.

```powershell
# Start the server in the first terminal
python -m web.app
```

```powershell
# Run an undefended attack in a second terminal
python -m attack.attacker --queries 500

# Run a defended attack
python -m attack.attacker --queries 500 --defended
```

Defended mode uses the defence settings currently configured in the Flask application. The attacker simulator does not replace those settings with fixed demonstration values.

Train copycat models separately:

```powershell
python -m training.train_substitute --mode undefended
python -m training.train_substitute --mode defended
```

## 8. Reproducible Experiment Records

Each HTTP attack run creates:

- A unique run ID such as `run001`
- An experiment UTC timestamp
- The requested and harvested query counts
- HTTP status counts
- Defended or undefended mode
- A snapshot of the active defence configuration
- Dataset SHA-256 hash
- Random seed
- Copycat architecture parameters
- Average server response latency
- Output dataset path

Stolen datasets are stored separately, for example:

```text
stolen_dataset_undefended_run001.csv
stolen_dataset_defended_run001.csv
```

Experiment history is exported to:

- `logs/experiment_runs.jsonl`
- `logs/experiment_runs.csv`

Security and prediction events are available through `logs/security_events.jsonl` and the `/api/security/stats` endpoint.

## 9. Evaluation Metrics

The project measures:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix
- Extraction fidelity
- Probability drift
- Harvested sample count
- HTTP 429 and 401 counts
- Perturbed-response count
- Average API latency
- Dataset hash and run metadata

> Fidelity reduction across repeated experiments at multiple query budgets is the strongest primary result. Individual copycat accuracy values should not be interpreted as the project’s main success metric.

## 10. Testing

Run the unit and integration suite:

```powershell
python -m unittest discover tests
```

The test suite covers preprocessing, health checks, Flask endpoints, rate limiting, suspicious-query detection, API security configuration, and deterministic perturbation behavior.

## 11. Important Evaluation Note

The project demonstrates that adaptive perturbation and request-rate controls can reduce copycat agreement, but it does not claim measured 100% genuine-user utility.

Utility claims should be supported by separate measurements of legitimate-user accuracy, perturbation rate, latency, false positives, and prediction agreement. Such user-profile evaluation is recommended before making quantitative claims about production usability.

## 12. Recommended Future Extensions

- Repeated experiments with confidence intervals and statistical summaries
- Query budgets of 200, 500, 1,000, and 2,000
- Multiple copycat architectures
- Genuine-user utility benchmark profiles
- True-positive and false-positive defense measurements
- Percentage of blocked and harvested requests
- Database-backed audit storage
- PDF and CSV report downloads
- Background experiment cancellation
- Docker deployment
- Automated browser testing

# Billitio-ML: Trip Reason Classification

A production-ready machine learning pipeline that classifies transportation booking trips as **Work** or **International** travel using Iranian ticket reservation data. Built with XGBoost, served via FastAPI, and includes an interactive web UI.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Experiment Notebooks](#experiment-notebooks)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
  - [API Server](#api-server)
- [API Reference](#api-reference)
- [Web Interface](#web-interface)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)

## Overview

Billitio-ML is an end-to-end ML system that predicts the **TripReason** (Work vs. International) for transportation bookings. The pipeline covers:

1. **Data Preprocessing** - Cleaning, feature engineering, encoding
2. **Model Training** - XGBoost classifier with threshold optimization
3. **Batch Prediction** - Generate submission files from test data
4. **REST API** - FastAPI server with interactive Persian-language web UI

## Project Structure

```
Billitio-ML/
├── api/
│   └── app.py                  # FastAPI server + embedded HTML UI
├── artifacts/                  # Trained model & preprocessing artifacts
│   ├── model.joblib            # Trained XGBoost classifier
│   ├── preprocessor.joblib     # Fitted preprocessing pipeline
│   └── metadata.json           # Label mappings, threshold, metrics
├── data/
│   ├── train_data.csv          # Training dataset (~101K rows)
│   ├── test_data.csv           # Test dataset (~43K rows)
│   ├── submission.csv          # Generated predictions
│   └── tarin_data.dev.csv      # Small development sample
├── notebooks/
│   ├── EDA.ipynb               # Exploratory Data Analysis
│   └── experiments/
│       ├── KNN.ipynb           # K-Nearest Neighbors experiment
│       ├── LogisticRegression.ipynb
│       ├── RandomForest.ipynb
│       ├── SVM.ipynb           # Support Vector Machine experiment
│       ├── XGBoost.ipynb       # Best model (selected)
│       ├── ROC_Curves.ipynb    # ROC curve comparison across models
│       └── Result.ipynb        # Final results summary
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point (train / predict)
│   ├── config.py               # Frozen configuration dataclass
│   ├── preprocessing.py        # Feature engineering & encoding pipeline
│   ├── modeling.py             # Training, splitting, threshold search
│   ├── model.py                # XGBoost model builder utility
│   ├── train.py                # Training pipeline orchestration
│   ├── predict.py              # Prediction pipeline orchestration
│   └── utils.py                # File I/O helpers (JSON, directories)
├── Makefile                    # Build automation commands
├── .gitignore
└── README.md
```

## Dataset

The dataset consists of Iranian transportation booking records with 21 columns:

| Column | Type | Description |
|--------|------|-------------|
| `TicketID` | ID | Unique ticket identifier |
| `BillID` | ID | Order/bill identifier (groups multiple tickets) |
| `UserID` | ID | User identifier |
| `NationalCode` | ID | National ID code |
| `HashPassportNumber_p` | Hash | Hashed passport number |
| `HashEmail` | Hash | Hashed email address |
| `BuyerMobile` | Hash | Buyer mobile number |
| `Created` | Datetime | Ticket creation timestamp |
| `CancelTime` | Datetime | Cancellation timestamp (if cancelled) |
| `DepartureTime` | Datetime | Departure timestamp |
| `Vehicle` | Categorical | Transport type (Bus, Train, Plane, InternationalPlane) |
| `VehicleType` | Categorical | Vehicle subtype |
| `VehicleClass` | Categorical | Service class (VIP, Economy, etc.) |
| `From` | Categorical | Origin city |
| `To` | Categorical | Destination city |
| `ReserveStatus` | Categorical | Reservation status |
| `Male` | Boolean | Passenger gender |
| `Domestic` | Boolean | Domestic trip flag |
| `Cancel` | Boolean | Cancellation flag |
| `Price` | Numerical | Ticket price (Iranian Rial) |
| `CouponDiscount` | Numerical | Discount coupon amount |
| **`TripReason`** | **Target** | **Trip purpose: `Work` or `Int` (International)** |

**Split:** ~101K training rows / ~43K test rows

## Feature Engineering

The preprocessing pipeline (`src/preprocessing.py`) creates the following engineered features:

### Group Features
- **`TicketPerOrder`** - Number of tickets per BillID (order size)
- **`family`** - Boolean: whether a BillID contains both male and female passengers

### Time Features
- **`Departure_Created`** - Days between ticket creation and departure (booking lead time)
- **`DepartureMonth`** - Month of departure

### Discount Features
- **`Discount`** - Boolean: whether a coupon discount was applied
- **`Price`** - Final price after subtracting coupon discount

### Data Cleaning
- **Boolean columns** (`Male`, `Domestic`, `Cancel`) converted to 0/1
- **`VehicleClass`** missing values filled with mode
- **Price outliers** removed via IQR method (k=10.0) on training data, clipped on test data
- **Invalid prices** (<=0) removed from training set

### Encoding
- **`Vehicle`** - OneHotEncoded (preserves category structure)
- **Other categoricals** (`VehicleClass`, `From`, `To`, `ReserveStatus`) - OrdinalEncoded with unknown handling
- **Target** (`TripReason`) - LabelEncoded (Int=0, Work=1)

### Dropped Columns
ID and hash columns are dropped as they carry no learning signal: `TicketID`, `UserID`, `HashPassportNumber_p`, `HashEmail`, `BuyerMobile`, `NationalCode`, `VehicleType`

**Final feature count: 18**

## Model Architecture

### Algorithm: XGBoost Classifier

| Hyperparameter | Value |
|----------------|-------|
| `n_estimators` | 800 |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 1.0 (L2 regularization) |
| `eval_metric` | logloss |
| Early stopping | 50 rounds (on validation set) |

### Key Design Decisions

- **GroupShuffleSplit** for train/validation split (80/20) - ensures all tickets from the same BillID stay in the same split, preventing data leakage
- **Threshold optimization** - instead of the default 0.5, the optimal threshold is searched in [0.05, 0.95] to maximize F1 score on validation data
- **Stateful preprocessing** - the `Preprocessor` class fits on training data and applies identical transformations to test data

## Experiment Notebooks

Multiple classifiers were evaluated during experimentation:

| Notebook | Algorithm | Purpose |
|----------|-----------|---------|
| `EDA.ipynb` | - | Exploratory Data Analysis & visualization |
| `KNN.ipynb` | K-Nearest Neighbors | Baseline comparison |
| `LogisticRegression.ipynb` | Logistic Regression | Linear baseline |
| `SVM.ipynb` | Support Vector Machine | Kernel-based approach |
| `RandomForest.ipynb` | Random Forest | Ensemble tree baseline |
| `XGBoost.ipynb` | XGBoost | **Selected model** |
| `ROC_Curves.ipynb` | - | ROC curve comparison across all models |
| `Result.ipynb` | - | Final results summary |

**XGBoost was selected** as the best-performing model based on F1 score comparison.

## Model Performance

Results on the 20% validation set (GroupShuffleSplit):

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 81.97% |
| **F1 Score (Work)** | 84.92% |
| **Macro F1** | 81.25% |
| **Weighted F1** | 81.68% |
| **Optimal Threshold** | 0.43 |

### Per-Class Performance

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| **Int** (International) | 85.9% | 70.7% | 77.6% | 8,878 |
| **Work** | 79.7% | 90.8% | 84.9% | 11,243 |

## Installation

### Prerequisites

- Python 3.12+
- pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Billitio-ML

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost scipy joblib fastapi uvicorn pydantic matplotlib
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=2.0 | Data manipulation |
| numpy | >=2.0 | Numerical operations |
| scikit-learn | >=1.5 | Preprocessing, splitting, metrics |
| xgboost | >=2.0 | Gradient boosting classifier |
| scipy | >=1.10 | Scientific computing |
| joblib | >=1.3 | Model serialization |
| fastapi | >=0.100 | REST API framework |
| uvicorn | >=0.20 | ASGI server |
| pydantic | >=2.0 | Request validation |
| matplotlib | >=3.8 | Visualization (notebooks) |

## Usage

### Training

Train the model and save artifacts:

```bash
# Using Make
make train

# Or directly
python3 -m src.main train --train_path data/train_data.csv
```

**CLI options:**

```
--train_path    Path to training CSV (required)
--out_dir       Output directory for artifacts (default: artifacts)
--test_size     Validation split ratio (default: 0.2)
--random_state  Random seed (default: 42)
```

**Output artifacts saved to `artifacts/`:**
- `preprocessor.joblib` - Fitted preprocessing pipeline
- `model.joblib` - Trained XGBoost model with threshold
- `metadata.json` - Label mappings, metrics, configuration

### Prediction

Generate predictions on test data:

```bash
# Using Make
make predict

# Or directly
python3 -m src.main predict \
  --test_path data/test_data.csv \
  --artifacts_dir ./artifacts \
  --output_path data/submission.csv
```

**CLI options:**

```
--test_path      Path to test CSV (required)
--artifacts_dir  Directory containing trained artifacts (default: artifacts)
--output_path    Output submission file path (default: artifacts/submission.csv)
```

**Output:** CSV file with columns `TicketID` and `TripReason`.

### API Server

Launch the FastAPI server:

```bash
# Using Make
make deploy-swagger

# Or directly
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

The server loads artifacts from the `ARTIFACTS_DIR` environment variable (default: `artifacts`).

## API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

### `POST /predict`

Predict trip reason for one or more records.

**Request:**
```json
{
  "records": [
    {
      "TicketID": 123,
      "BillID": 999,
      "Price": 250000,
      "CouponDiscount": 0,
      "Vehicle": "Bus",
      "VehicleClass": "VIP",
      "Created": "2025-01-10T10:00",
      "DepartureTime": "2025-01-12T08:00",
      "Male": true,
      "Domestic": true,
      "Cancel": false
    }
  ]
}
```

**Response:**
```json
{
  "predictions": ["Work"],
  "probabilities": [0.78],
  "threshold": 0.43
}
```

### `GET /`

Serves the interactive web UI.

### `GET /docs`

Auto-generated Swagger/OpenAPI documentation (provided by FastAPI).

## Web Interface

The API includes an embedded interactive web UI accessible at `http://localhost:8000/`.

**Features:**
- Persian (Farsi) language interface with RTL layout
- Pre-filled default fields with example values
- Dynamic custom field addition (text, number, boolean types)
- Live JSON payload preview
- Copy JSON to clipboard
- Real-time prediction results display
- Dark theme with glassmorphism design (Tailwind CSS)

## Configuration

The `Config` dataclass (`src/config.py`) controls pipeline behavior:

```python
@dataclass(frozen=True)
class Config:
    random_state: int = 42
    test_size: float = 0.2
    drop_cols: tuple = (
        "TicketID", "UserID", "HashPassportNumber_p",
        "HashEmail", "BuyerMobile", "NationalCode", "VehicleType",
    )
    target_col: str = "TripReason"
    group_col: str = "BillID"
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12 |
| ML Framework | XGBoost, scikit-learn |
| Data Processing | pandas, NumPy |
| API | FastAPI + Uvicorn |
| Frontend | Tailwind CSS (embedded HTML) |
| Serialization | joblib |
| Automation | Make |
| Version Control | Git |


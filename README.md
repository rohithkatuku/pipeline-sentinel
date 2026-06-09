# 🛡️ Pipeline Sentinel

**AI-powered data pipeline health monitor & root cause analyzer.**

Pipeline failures silently corrupt downstream analytics, ML models, and business decisions. Engineering teams waste 30-40% of time debugging pipeline breaks manually. Pipeline Sentinel catches anomalies in real time, diagnoses root causes using LLMs, and maps blast radius across your data lineage.

## Features

- **Anomaly Detection** — Statistical + ML models (Z-score, IQR, Isolation Forest) monitor pipeline metrics: row counts, schema drift, data freshness, distribution shifts, null spikes
- **LLM Root Cause Analysis** — When anomaly fires, agent correlates across logs/metadata/lineage and generates plain-English diagnosis
- **Auto-Remediation Suggestions** — Ranked fix actions based on incident patterns
- **Lineage-Aware Impact Analysis** — Traces which downstream dashboards/models are affected
- **Pipeline Simulator** — Synthetic failure injection for demo and testing

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Data        │────▶│  Sentinel    │────▶│  Alert &        │
│  Pipelines   │     │  Detectors   │     │  Dashboard      │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │
                    ┌──────▼───────┐
                    │  LLM Root    │
                    │  Cause       │
                    │  Analyzer    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Lineage     │
                    │  Impact Map  │
                    └──────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI |
| Anomaly Detection | scikit-learn, scipy |
| Root Cause Analysis | Anthropic Claude API |
| Data Validation | Great Expectations-style checks |
| Metadata Store | SQLite |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |

## Quick Start

```bash
# Clone
git clone https://github.com/rohithkatuku/pipeline-sentinel.git
cd pipeline-sentinel

# Install
pip install -r requirements.txt

# Initialize database
python -m sentinel.models.database

# Run simulator (generates synthetic pipeline data + failures)
python -m simulations.pipeline_simulator

# Launch API server
uvicorn sentinel.api.main:app --reload

# Launch dashboard (separate terminal)
streamlit run dashboard/app.py
```

## Project Structure

```
pipeline-sentinel/
├── sentinel/
│   ├── detectors/        # Anomaly detection models
│   │   ├── statistical.py    # Z-score, IQR detectors
│   │   ├── ml_detector.py    # Isolation Forest detector
│   │   └── schema_drift.py   # Schema change detection
│   ├── analyzers/        # LLM root cause engine
│   │   └── root_cause.py
│   ├── validators/       # Data quality checks
│   │   └── quality.py
│   ├── lineage/          # Dependency graph tracker
│   │   └── tracker.py
│   ├── api/              # FastAPI endpoints
│   │   └── main.py
│   └── models/           # Database models
│       └── database.py
├── dashboard/            # Streamlit monitoring UI
│   └── app.py
├── simulations/          # Synthetic pipeline data generator
│   └── pipeline_simulator.py
├── tests/
│   ├── test_detectors.py
│   ├── test_validators.py
│   └── test_lineage.py
└── docs/
    └── design.md
```

## Environment Variables

```bash
ANTHROPIC_API_KEY=your_key_here   # For LLM root cause analysis
SENTINEL_DB_PATH=sentinel.db      # SQLite database path
```

## License

MIT

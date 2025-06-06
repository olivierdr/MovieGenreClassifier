# Movie Genre Classifier

A machine learning project for classifying movies into genres based on their synopses.

## Project Structure

```
MovieGenreClassifier/
├── data/
│   ├── raw/           # Original, immutable data
│   └── processed/     # Cleaned and processed data
├── outputs/
│   ├── analysis/      # Data analysis results and plots
│   ├── logs/         # Log files for data processing and model training
│   └── models/       # Trained models and model artifacts
├── src/
│   ├── data/         # Data processing scripts
│   │   ├── analysis.py    # Data exploration and analysis
│   │   └── clean_dataset.py  # Data cleaning pipeline
│   ├── models/       # Model training and evaluation
│   │   ├── model.py      # Model definitions
│   │   └── train.py      # Training pipeline
│   ├── utils/        # Utility functions
│   └── app.py        # Streamlit web application
├── tests/            # Test files
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline

The project follows this pipeline:

1. Data Analysis and Cleaning:
```bash
# Run data analysis
python src/data/analysis.py

# Run data cleaning
python src/data/clean_dataset.py
```
This will:
- Analyze the raw data and generate plots in `outputs/analysis/`
- Clean the data and save processed files in `data/processed/`
- Generate logs in `outputs/logs/`

2. Model Training:
```bash
python src/models/train.py
```
This will:
- Train both TF-IDF and Embedding models
- Save models in `outputs/models/`
- Generate training logs in `outputs/logs/`

3. Web Application:
```bash
streamlit run src/app.py
```
This will:
- Start a web interface for movie genre prediction
- Use both trained models for predictions

## Development

- Use `src/` directory for source code
- Add tests in `tests/` directory
- Log files are stored in `outputs/logs/`
- Models are saved in `outputs/models/`
- Analysis results and plots are in `outputs/analysis/`

## Data Files

- Raw data: `data/raw/task.csv`
- Processed data:
  - `data/processed/movies_cleaned.csv`: Clean dataset for training
  - `data/processed/movies_checked_ambiguous.csv`: Full dataset with ambiguity flags
  - `data/processed/movies_ambiguous.csv`: Only ambiguous entries

## License

[To be added: License information]
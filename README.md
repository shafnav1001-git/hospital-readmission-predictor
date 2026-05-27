# Hospital Readmission Predictor

A machine learning project that predicts whether a patient will be readmitted to hospital within 30 days of discharge. Built to practice data preprocessing, model training, and building a simple API.

## What it does
Takes basic patient info (age, diagnosis, length of stay, number of medications) and returns a readmission risk prediction.

## Tech Stack
- Python, Pandas, scikit-learn
- Flask (REST API)

## Project Structure
```
├── notebooks/       # EDA and model training
├── src/             # Preprocessing and prediction scripts
├── api/app.py       # Flask API
└── models/          # Saved trained model
```

## Getting Started

```bash
git clone https://github.com/your-username/hospital-readmission-predictor.git
cd hospital-readmission-predictor
pip install -r requirements.txt

# Train the model
python src/train.py

# Run the API
python api/app.py
```



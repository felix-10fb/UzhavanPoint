# Crop Recommendation API

This workspace contains a small FastAPI backend that serves crop recommendations from a Random Forest model.

Files:
- `train.py` - trains a RandomForest model and saves `crop_recommendation.pkl` (if `Crop_recommendation.csv` is present it uses it; otherwise it generates dummy data).
- `main.py` - FastAPI app that loads `crop_recommendation.pkl` (will run `train.py` automatically if the pickle is missing) and exposes `/predict_crop`.

Quick start

1. Create a Python virtual environment and activate it (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the model (optional — `main.py` will run training if pickle is missing):

```powershell
python train.py
```

4. Run the API with Uvicorn:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

5. Example request (use `curl` or Postman):

```bash
curl -X POST "http://127.0.0.1:8000/predict_crop" -H "Content-Type: application/json" -d \
'{"N":90,"P":42,"K":43,"temperature":30.0,"humidity":75.0,"ph":6.5,"rainfall":200.0}'
```

Notes
- If you have the Kaggle `Crop_recommendation.csv` dataset, place it in the workspace root before running `train.py` to train on the real dataset.
- `train.py` already saves the model as `crop_recommendation.pkl` which `main.py` consumes.

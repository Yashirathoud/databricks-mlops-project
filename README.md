### In your local machine's project folder:
```powershell
.
├── .github/
│   └── workflows/
│       ├── infer.yml
│       └── train.yml
├── notebooks/  <-- CREATE THIS FOLDER
│   ├── 01_model_training.py  <-- PLACE YOUR TRAINING CODE HERE
│   └── 02_batch_inference.py <-- PLACE YOUR INFERENCE CODE HERE
└── requirements.txt

```

### Python environment:
```powershell
 python -m venv .venv
.venv\Scripts\activate.bat

```

pip install -r requirements.txt

## Temporarily allow scripts (recommended for local dev)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv\Scripts\Activate.ps1


python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
 databricks configure --token

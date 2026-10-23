# Skin Cancer Classification Application

## 1. Project Overview

This project is a skin lesion classification research prototype developed using Streamlit and FastAPI. It uses ResNet-50 for feature extraction and an SVM classifier to generate predictions. The project also explores fairness in skin cancer classification across different skin tones.

**Disclaimer:** This application is for educational and research purposes only. It is not a medical diagnostic tool.

## 2. Requirements

- Windows 10/11
- Python
- Internet connection for installing dependencies

### Install Python

1. Download Python from the official website: https://www.python.org/downloads/windows/
2. Run the installer and **tick “Add python.exe to PATH”** before selecting Install Now.
3. Open PowerShell and verify the installation:

```powershell
python --version
python -m pip --version
```

If these commands fail, restart PowerShell and try again.

## 3. Project Structure

```text
Skin Cancer Project/
├── README.md
├── Final_SkinCancer_API/
│   ├── main.py
│   ├── Requirements.txt
│   ├── ada_pca.pkl
│   ├── ada_scaler.pkl
│   └── ada_svc.pkl
└── skin cancer app/
    ├── app.py
    ├── requirements.txt
    └── evaluation_files/
```

Keep the folder structure and filenames unchanged.

## 4. Run the Backend API

Open PowerShell in the main project folder and install the backend dependencies:

```powershell
python -m pip install -r ".\Final_SkinCancer_API\Requirements.txt"
```

Start the API:

```powershell
cd ".\Final_SkinCancer_API"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Keep this terminal open. Check the API at http://127.0.0.1:8000/docs.

## 5. Run the Streamlit Application

Open a **second PowerShell window** in the main project folder.

Install the frontend dependencies:

```powershell
python -m pip install -r ".\skin cancer app\requirements.txt"
```

Start the application:

```powershell
cd ".\skin cancer app"
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually http://localhost:8501.

Keep both terminals running while using the app.

## 6. Notes

- Both the backend API and frontend must be running.
- If Python is not recognised, check the installation and PATH settings.
- Ensure all model files and evaluation files are included.
- Model predictions and certainty scores do not guarantee accuracy. Fairness conclusions should be based on verified evaluation results.
- This project is a research prototype and must not be used to make medical decisions.
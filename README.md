# 🚗 Turn Signal Dataset Project

This repository contains code and documentation for creating a dataset of vehicle light states (left/right turn signals, hazard, brake, tail lights).  

---

## 📦 Project Setup

### Create and Activate a Python Environment

I use `pyenv` to manage Python versions and virtual environments.

```
# Install and select Python version (if not already installed)
pyenv install 3.12.6
pyenv virtualenv 3.12.6 adl
pyenv local adl
```

### Install dependencies

```
pip install -r requirements.txt
```

### Directory Structure

```
turn-signal-dataset/
├── notebooks/              # Jupyter notebooks for exploration
├── sampled_images/            # Local sampled images (ignored by Git)
├── .env                    # API keys (not tracked by Git)
├── .gitignore
├── requirements.txt
└── README.md
```

### API Key Setup

1. Create a .env file in the project root
2. Add: OPENAI_API_KEY=sk-your-key-here
3. Ensure .env is in .gitignore 

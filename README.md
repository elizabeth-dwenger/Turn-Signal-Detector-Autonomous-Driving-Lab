# 🚗 Turn Signal Dataset Project

This repository contains code and documentation for creating a dataset of vehicle light states (left/right turn signals, hazard, tail light, tail light off, and no visible tail lights).  

---

## Project Setup

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
ADL/
├── jupyter/              # Jupyter notebooks for exploration
├── sampled_images/            # Local sampled images (ignored by Git)
├── front_back_images/         # Local sampled images prior to filtering
├── .env                    # API keys (not tracked by Git)
├── .gitignore
├── requirements.txt
└── README.md
```

### API Key Setup

1. Create a .env file in the project root
2. Add: OPENAI_API_KEY=sk-your-key-here
3. Ensure .env is in .gitignore 

### JSON annotations format

```
[
  {
    "image": "../data_sample/car_001.jpg",
    "turn_signal": "left",
    "tail_light": "on"
  },
  {
    "image": "../data_sample/car_002.jpg",
    "turn_signal": "none",
    "tail_light": "no_visible"
  },
  {
    "image": "../data_sample/car_003.jpg",
    "turn_signal": "both",
    "tail_light": "off"
  }
]
```

---
# lauetools-utils

This README explains step-by-step how to create a **reproducible environment** for LaueTools + `lauetools-utils`, connect it to Jupyter, and run a quick test.  
It is written for users with little experience in Python, bash, or conda.

---
## 1. Installation

## 2. Set up in a virtual environment

### 2.1 Conda

Check if Conda is installed (install only if needed)

First check if `conda` is already installed:

```bash
conda --version
```

✅ If you see a version (e.g. conda 24.5.0), skip to Step 1.

❌ If you get command not found, install Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda init bash
source ~/.bashrc
conda --version   # check installation again
```
This installs Miniconda in your home directory.

---

#### Install dependencies through `environment.yml`
If you cloned the repository, you should have the environment.yml file in the project root (same folder where laueutils was cloned from git). Then run:

```bash
conda env create -f environment.yml
conda activate lauetools-utils-env
```

#### Updating the environment
Only run this command if dependencies have changed in environment.yml or pyproject.toml:

```bash
conda env update -f environment.yml --prune
``` 
If nothing changed, you do not need to run this again.

### 2.2 Venv

Alternative: Create environment with venv (no conda)
If you cannot use conda, you can still create a virtual environment with Python’s built-in venv:

```bash
python3 -m venv .venv
source .venv/bin/activate  # (Linux/Mac)
# On Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
---

### 2.3 Add the kernel to Jupyter (optional) 
With the environment activated (conda activate lauetools-utils-env or source .venv/bin/activate):

```bash
python -m ipykernel install --user --name lauetools-utils-env --display-name "Python (lauetools-utils)"
This makes the environment appear in Jupyter Notebook/Lab.
```
---

### 2.4 Install lauetools-utils (editable mode)
Make sure you are in the repository root (use pwd to check, you should see pyproject.toml there). Then run:

```bash
pip install -e .
```

This installs the package in editable mode, so code changes take effect immediately.

---

### 2.5 Run a test – Jupyter Notebook
Open JupyterLab/Notebook, select the Python (lauetools-utils) kernel and run:

```python
import numpy, scipy, matplotlib, h5py, fabio
import laueutils
print("✅ Environment + laueutils import OK")
```

If you see the checkmark message without errors, your environment is ready!
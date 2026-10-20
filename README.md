# AI Programming Foundations Project

This project is a reproducible data workflow in a Jupyter notebook (`data_workflow.ipynb`). It loads the Titanic passenger data, cleans it with documented functions, explores survival patterns and saves three labeled figures to `figures/`. Dataset: Titanic - Machine Learning from Disaster (Kaggle), https://www.kaggle.com/c/titanic/data. The file `titanic.csv` (Kaggle `train.csv`, 891 rows x 12 columns) is included in the repository.

## How to run

1. Clone the repository and open the folder:
   ```bash
   git clone https://github.com/Sirvanmesh/ai-programming-foundations-project.git
   cd ai-programming-foundations-project
   ```
2. Create and activate a virtual environment (Python 3.9 or newer; tested with 3.11):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS / Linux
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Jupyter and open `data_workflow.ipynb`:
   ```bash
   jupyter notebook
   ```
5. Run all cells (`Kernel > Restart Kernel and Run All Cells`). The figures are written to `figures/`.

To run the notebook from the command line instead:
```bash
jupyter nbconvert --to notebook --execute --inplace data_workflow.ipynb
```

## How requirements.txt was created

The file was generated from the project virtual environment with:
```bash
pip freeze > requirements.txt
```

## Files

- `data_workflow.ipynb` - the executed notebook
- `titanic.csv` - the dataset
- `requirements.txt` - pinned package versions
- `figures/` - Figures 1-3 (PNG)
- `module_summary.pdf` - written report

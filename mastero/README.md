# MASTERO - Master's Thesis Experimental Framework

This directory contains the complete experimental framework for the master's thesis research on "Enhancing Symbolic Regression-Based GSGP for Binary Classification" using SLIM-GSGP algorithms.

## 📁 Directory Structure

```
mastero/
├── data/                           # Dataset management and results
│   ├── data_info.csv              # Dataset metadata and train/test splits
│   ├── data_original/             # Raw datasets from UCI/OpenML
│   ├── data_prepared/             # Preprocessed datasets ready for experiments
│   └── results/                   # Experimental results and logs
│       ├── RQ_Fitness/            # Experiment 1: Fitness function analysis
│       ├── RQ_Inflationrate/      # Experiment 2: SLIM parameter analysis
│       └── RQ_Comparison/         # Experiment 3: Algorithm comparison
├── data_preprocessing/            # Data preparation pipeline
│   ├── data_preparation.ipynb    # Jupyter notebook for data preprocessing
│   └── utils.py                  # Preprocessing utility functions
└── experiments/                  # Core experimental framework
    ├── analysis.py               # Statistical analysis and visualization
    ├── experiment.py             # Main experiment orchestration
    ├── monte_carlo.py            # Monte Carlo cross-validation implementation
    ├── utils.py                  # Experiment utility functions
    ├── basic_model_config.py     # Base model configurations
    ├── RQ_Fitness/              # Experiment 1 setup and analysis
    ├── RQ_Inflationrate/        # Experiment 2 setup and analysis
    ├── RQ_Comparison/           # Experiment 3 setup and analysis
    └── Latex/                   # Auto-generated LaTeX tables and figures
```

## 🧪 Research Experiments

This framework implements three key research questions:

### **Experiment 1: Fitness Function Analysis (RQ_Fitness)**
- **Objective**: Evaluate the impact of different fitness functions on GSGP performance
- **Methods**: Compares RMSE, WRMSE, Accuracy, and F1-Score with different activation functions
- **Grid Search**: 4 fitness functions × 4 mutation upper steps = 16 configurations
- **Key Focus**: Regression-based vs. classification-based fitness functions

### **Experiment 2: SLIM Parameter Analysis (RQ_Inflationrate)**  
- **Objective**: Analyze inflation rate and mutation step effects in SLIM-GSGP
- **Methods**: Tests SLIM+SIG1 and SLIM*SIG1 variants across parameter ranges
- **Grid Search**: 2 SLIM variants × 4 inflation rates × 4 mutation steps = 32 configurations
- **Key Focus**: Performance-complexity trade-off optimization

### **Experiment 3: Algorithm Comparison (RQ_Comparison)**
- **Objective**: Compare SLIM-GSGP, traditional GSGP, and standard GP
- **Methods**: Head-to-head performance and model complexity comparison
- **Key Focus**: Validation of SLIM-GSGP benefits for binary classification

## 📊 Datasets

**10 Binary Classification Benchmark Datasets:**
- Auction, Autism, Biomed, Credit, Darwin, Heart, PC4, Thyroid, Wilt, Wisconsin
- Sources: UCI Machine Learning Repository, OpenML platform
- Preprocessing: Z-score standardization, one-hot encoding, class balancing

## ⚙️ Experimental Setup

### **Statistical Methodology**
- **Cross-Validation**: 30 Monte Carlo runs with 70/30 train/test splits
- **Statistical Tests**: Wilcoxon rank-sum tests, Friedman tests (α = 0.05)
- **Analysis**: Win-tie-loss tables, ranking analysis, effect size evaluation

### **Hyperparameters**
```python
Population Size: 100
Generations: 1000  
Tournament Size: 2
Initialization: Ramped Half-and-Half (depth 6)
Functions: [add, subtract, multiply, divide]
Constants: [-10, 10] (201 values)
```

### **SLIM-GSGP Variants**
- **SLIM+SIG1**: Addition-based with single sigmoid function
- **SLIM*SIG1**: Multiplication-based with single sigmoid function  
- **Mathematical Forms**:
  - SLIM+SIG1: `GSI(T) = T + ms·(2·σ(T_R) - 1)`
  - SLIM*SIG1: `GSI(T) = T·(1 + ms·(2·σ(T_R) - 1))`

## 🚀 Quick Start

### 1. **Run Single Experiment**
```bash
cd experiments/RQ_Fitness
python setup.py --experiment_name "MyExperiment" --dataset_name "auction"
```

### 2. **Run Full Analysis**
```python
from analysis import FitnessAnalysis
ana = FitnessAnalysis('RQ_Fitness')
# Automatically generates statistical tables and plots
```

### 3. **View Results**
- Results stored in: `data/results/{experiment_name}/{dataset}/`
- LaTeX outputs in: `experiments/Latex/Chapters/`
- Analysis notebooks in: `experiments/{experiment}/`

## 📈 Analysis Framework

### **Statistical Analysis Classes**
```python
Analysis()              # Base analysis functionality
├── FitnessAnalysis()   # Experiment 1 statistical processing  
├── InflationrateAnalysis()  # Experiment 2 parameter analysis
└── ComparisonAnalysis()     # Experiment 3 algorithm comparison
```

### **Key Analysis Features**
- **Automated Statistical Testing**: Wilcoxon, Friedman tests with proper multiple comparisons
- **Win-Tie-Loss Analysis**: Pairwise significance testing across all configurations
- **Ranking Systems**: Performance ranking with statistical validation
- **Trade-off Analysis**: Multi-objective optimization using weighted Euclidean distance
- **LaTeX Generation**: Publication-ready tables and figures

### **Performance Metrics**
- **Classification**: Accuracy, F1-Score, ROC-AUC, Precision, Recall
- **Regression**: RMSE, Weighted RMSE (with class balancing)
- **Complexity**: Tree node count, semantic space analysis

## 🔧 Key Implementation Features

### **Data Preprocessing Pipeline**
- **Standardization**: Z-score normalization (fit on train, apply to test)
- **Categorical Encoding**: One-hot encoding for categorical features
- **Class Encoding**: Majority class → 0, Minority class → 1
- **Missing Data**: Removal of incomplete samples and ID columns

### **Monte Carlo Framework**
- **Reproducible**: Consistent seed management across all experiments  
- **Scalable**: Parallel execution support for multiple datasets
- **Robust**: Complete error handling and progress tracking
- **Configurable**: Flexible model configuration system

### **Result Storage System**
- **Structured CSV**: Complete metrics for each run and configuration
- **Detailed Logs**: Generation-by-generation evolution tracking
- **Configuration Tracking**: Full hyperparameter preservation
- **Metadata Management**: Dataset information and experimental settings

## 🎯 Research Contributions

This framework enables the investigation of:

1. **Fitness Function Impact**: First systematic comparison of regression vs. classification-based fitness functions in GSGP
2. **SLIM-GSGP for Classification**: Novel application of SLIM-GSGP to binary classification problems
3. **Parameter Optimization**: Comprehensive analysis of inflation rate and mutation step effects
4. **Performance-Complexity Trade-offs**: Multi-objective optimization in GP model selection
# Structure of the project
```
/TSPytorch
│
│
├── models
│   ├── __init__.py
│   ├── Informer.py
│   ├── Autoformer.py
│   ├── FEDformer.py
│   ├── ...
│   └── New_Model.py
│
├── utils
│   ├── __init__.py
│   ├── data.py
│   └── metrics.py
│
├── config
│
│
├── layers # inspired from Autoformer
│   ├── __init__.py
│   ├── AutoCorrelation.py
│   ├── Transformer_EncDec.py
│   ├── utils.py
│   └── ...
│
├── dataset
│   ├── ETT
│   ├── electricity
│   ├── __init__.py
│   └── data_loader.py
│
├── scripts # 2 types: script to experiments and deployments
│   ├── experiments  
│   └── deployments
│
├── notebooks
│   ├── exploratory_data_analysis.ipynb
│   ├── model_evaluation.ipynb
│   └── ...
│
├── model_saved
│   ├── checkpoints # model checkpoints:  *.pth, log.txt
│   │       ├── model1
│   │       ├── model2
│   │       └── ...
│   └── results # model evaluation results: metrics, figures, pred.npy, ...
│           ├── model1
│           ├── model2
│           └── ...
│
│
│
├── main.py  # final script to run publicly -> scripts/deployments/
│
├── experiment.py # script to run experiments -> scripts/experiments/
│
├── README.md
├── requirements.txt
├── environment.yml
├── Dockerfile
└── Makefile

```

## Explanation of each part
- `config/`: having a config folder with several configuration files can be useful to organize different sets of parameters, possibly
for different environments (like development, testing, and production).
- `model_saved`/: This is where you should store all your experimental outputs, including model parameters, output data, logs, and other related items. Each experiment has its own folder.
- `notebooks`/: This is where you should put all your Jupyter notebooks. These can be used for exploratory data analysis, data visualization, model evaluation, etc.
- `requirements.txt`, `environment.yml`: This file should list all the Python packages that your project depends on.
- `Dockerfile`: This file contains all the commands a user could call on the command line to assemble an image.
- `Makefile`: This file contains a set of directives used by a make build automation tool to generate a target/goal.









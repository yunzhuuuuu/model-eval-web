### Repo Structure
cori_model_eval/
├── evaluation.py
├── visualization.py
├── datasets/
├── embeddings/

### How to run
1. download datasets and embeddings
2. install dependencies
```pip install -r requirements.txt
```
3. run streamlit
```
streamlit run visualization.py
```

### TODO:
- script to convert csv dataset to npz
- write instructions
- host the web and figure out how students submit their dataset (in csv?) and write/choose evaluation metrics?
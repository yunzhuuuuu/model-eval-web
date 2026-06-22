### Repo Structure
cori_model_eval/  
├── evaluation.py  
├── visualization.py  
├── datasets/  
├── embeddings/  

### How to run
1. download datasets and embeddings from: https://drive.google.com/drive/folders/1h8fCsoqWcUGh9Cg94q-NLVB95tbExlXx?usp=sharing
and put them under the root folder
2. install dependencies
```
pip install -r requirements.txt
```
3. run streamlit
```
streamlit run visualization.py
```

### TODO
- figure out how students submit their dataset (in csv?) through the web
- write a script to convert csv dataset to npz
- write instructions
- deploy the web
- students write/choose their own evaluation metrics? **(ask cori if she wants this)**
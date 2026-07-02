### Repo Structure
cori_model_eval/  
├── evaluation.py  
├── app.py  
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
streamlit run app.py
```

### TODO
- figure out how students submit their dataset/api keys through the web
- write a script to convert csv dataset to embeddings
    - sentence embeddings done, need to work on gemini
- write instructions
    - the students should submit topic name, facts.csv, qanda.csv
- deploy the web
- modulize the three sections
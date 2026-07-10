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

#### How to get Gemini API key
If you want to test gemini model, you should set up your own API key and enable generate_gemini_embeddings for embed_dataset().
1. Visit https://aistudio.google.com/api-keys and create an API key.
2. Create a .env file with:
```
GEMINI_API_KEY=your_key_here
```
and copy paste your api key here.

### TODO
- figure out how students submit their dataset/api keys through the web
- write a script to convert csv dataset to embeddings
    - sentence embeddings done, need to work on gemini
- deploy the web
- modulize the three sections
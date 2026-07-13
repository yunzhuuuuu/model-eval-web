### Repo Structure
cori_model_eval/  
├── app.py  
├── generate_embeddings.py
├── evaluation.py  
├── datasets/  
├── embeddings/  
├── data_and_embeddings.tar.gz

### How to run
1. install dependencies
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
2. Create a .streamlit/secrets.toml file (need to update this)
```
GEMINI_API_KEY="your_key_here"
```
and copy paste your api key here.

### TODO
- figure out how to store and show students' datasets
- deploy the web
- modulize the three sections
### Repo Structure
cori_model_eval/  
├── app.py  
├── generate_embeddings.py
├── evaluation.py  
├── datasets/  
├── embeddings/  
├── data_and_embeddings.tar.gz

### How to run locally
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

To let it run on website, the key needs to be updated so streamlit.
1. Go to share.streamlit.io and log in
2. Click the "⋮" menu next to your app → Settings → Secrets
3. Paste in the same TOML content as the local file

### TODO
- modulize the three sections
- debug messages if the dataset files breaks

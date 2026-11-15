import json,os
from pathlib import Path
import joblib
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from .pipeline import MODELS,DATA,CATEGORIES

app=FastAPI(title='AgentFinder Machine Learning API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['GET','POST'],allow_headers=['*'])
class Query(BaseModel):text:str=Field(min_length=3,max_length=1000);limit:int=Field(default=5,ge=1,le=10)

def load(filename):
 p=MODELS/filename
 if not p.exists():raise HTTPException(status_code=503,detail='Model not found. Run Colab notebook crawling/training first and copy models + data to this folder.')
 return joblib.load(p)

@app.get('/api/health')
def health():return {'ready':all((MODELS/f).exists() for f in ('knn.joblib','naive_bayes.joblib','decision_tree.joblib'))}

@app.get('/api/catalog')
def catalog():
 import pandas as pd
 p=DATA/'agents_clean.csv'
 if not p.exists():return {'items':[],'count':0}
 df=pd.read_csv(p).fillna('');return {'items':df[['name','description','category','stars','language','github_url','website_url','source_url']].to_dict('records'),'count':len(df)}

@app.get('/api/metrics')
def metrics():
 p=MODELS/'metrics.json'
 if not p.exists():return {'available':False}
 return {'available':True,**json.loads(p.read_text(encoding='utf-8'))}

@app.post('/api/recommend')
def recommend(q:Query):
 from sklearn.metrics.pairwise import cosine_similarity
 b=load('knn.joblib');v=b['vectorizer'].transform([q.text]);items=b['catalog']
 # nearest neighbors with cosine metric; zero-similarity matches are discarded
 d,idx=b['model'].kneighbors(v,n_neighbors=min(q.limit,len(items)))
 recommendations=[]
 for distance,i in zip(d[0],idx[0]):
  score=max(0,1-float(distance))
  if score<=0:continue
  rec=items[int(i)];recommendations.append({key:rec.get(key,'') for key in ['name','description','category','stars','language','github_url','website_url','source_url'] }|{'similarity':round(score,4)})
 preds={}
 for key in ['naive_bayes','decision_tree']:
  model=load(key+'.joblib');preds[key]=str(model.predict([q.text])[0])
 return {'query':q.text,'predictions':preds,'recommendations':recommendations,
         'disclaimer':'Similarity is not a probability of suitability. Predictions use weak-label training data.'}

"""AgentFinder: crawl public GitHub project documentation, label, train and evaluate.
Only repositories with public GitHub API metadata and an accessible raw README are included.
Labels are provisional rules; inspect data/label_audit.csv before academic use.
"""
from __future__ import annotations
import os, re, time, json, csv, io
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import requests
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; MODELS=ROOT/'models'
for d in (DATA,MODELS): d.mkdir(exist_ok=True,parents=True)
HEADERS={'Accept':'application/vnd.github+json','User-Agent':'AgentFinder-Educational-Research'}
CATEGORIES=['Coding Agent','Multi-Agent Terminal','Agent Orchestrator']
QUERIES=['ai coding agent in:name,description,readme stars:>5',
         'coding agent terminal multiplexer in:name,description,readme stars:>2',
         'multi agent coding orchestration in:name,description,readme stars:>2',
         'parallel coding agents in:name,description,readme stars:>2']
SEEDS=['Aider-AI/aider','block/goose','All-Hands-AI/OpenHands','anomalyco/opencode',
       'smtg-ai/claude-squad','BloopAI/vibe-kanban','Dicklesworthstone/claude_code_agent_farm',
       'paddo/claude-setup','microsoft/autogen','crewAIInc/crewAI']
# Conservative lexical cues. Assignment is a reproducible weak-label classification study.
KEYWORDS={
 'Multi-Agent Terminal':['terminal multiplexer','terminal panes','split terminal','split panes','tmux sessions','manage multiple terminal','terminal tabs','multiple claude sessions','multiple coding sessions','agent sessions in parallel'],
 'Agent Orchestrator':['orchestrate agents','agent orchestration','multi-agent orchestration','multi agent workflow','coordinate multiple agents','agent task delegation','delegate tasks to agents','agent team','multi-agent framework','multi agent framework','workflow orchestration'],
 'Coding Agent':['coding agent','ai pair programmer','ai coding assistant','agentic coding','terminal coding assistant','software engineering agent','code editing agent','autonomous coding','codebase assistant','edit code','write code']}

def api_get(url,token=None,params=None):
 h=dict(HEADERS)
 if token: h['Authorization']='Bearer '+token
 try:
  r=requests.get(url,headers=h,params=params,timeout=25)
  if r.status_code in (403,429):
   raise RuntimeError(f'GitHub rate limit or permission error ({r.status_code}). Try GITHUB_TOKEN and rerun: {r.text[:180]}')
  if r.status_code==404:return None
  r.raise_for_status(); return r.json()
 except requests.RequestException as exc:
  print('API warning:',str(exc)[:160]);return None

def raw_readme(full_name,default_branch,token=None):
 """Use GitHub's public README API with raw markdown Accept header."""
 h=dict(HEADERS);h['Accept']='application/vnd.github.raw+json'
 if token:h['Authorization']='Bearer '+token
 try:
  r=requests.get(f'https://api.github.com/repos/{full_name}/readme',headers=h,timeout=20)
  if r.status_code!=200:return '',r.status_code
  # GitHub returns raw README with this media type
  t=r.text
  if t.lstrip().startswith('{'):
   try:
    import base64
    j=r.json();t=base64.b64decode(j.get('content','')).decode('utf-8','replace')
   except Exception:return '',r.status_code
  return t[:45000],200
 except requests.RequestException:return '',0

def plain(markdown):
 markdown=re.sub(r'```[\s\S]*?```',' ',markdown)
 markdown=re.sub(r'!\[[^]]*\]\([^)]*\)',' ',markdown)
 markdown=re.sub(r'\[([^]]+)\]\([^)]*\)',r'\1',markdown)
 markdown=re.sub(r'<[^>]+>',' ',markdown)
 markdown=re.sub(r'https?://\S+',' ',markdown)
 return re.sub(r'\s+',' ',markdown).strip()

def rule_label(text):
 t=text.lower();scores={k:sum(1 for phrase in phrases if phrase in t) for k,phrases in KEYWORDS.items()}
 # Specialized multi-agent keywords are stronger than generic 'coding agent'.
 weighted={k:(v*2 if k!='Coding Agent' else v) for k,v in scores.items()}
 best=max(weighted,key=weighted.get)
 ordered=sorted(weighted.values(),reverse=True)
 # Ambiguous cases are excluded from training; inspect label_audit.csv.
 if ordered[0]<1 or ordered[0]==ordered[1]:return 'REVIEW',scores
 return best,scores

def crawl(token=None,per_query=20,max_repos=85,delay=0.25):
 """Collect new data, never fabricate records or silently substitute downloaded datasets."""
 seen={}; logs=[]; buckets=[[] for _ in QUERIES]
 for name in SEEDS:
  repo=api_get('https://api.github.com/repos/'+name,token)
  if repo and not repo.get('private') and not repo.get('archived'):seen[repo['full_name'].lower()]=repo
  time.sleep(delay)
 for qi,q in enumerate(QUERIES):
  for page in (1,2):
   payload=api_get('https://api.github.com/search/repositories',token,{'q':q,'sort':'stars','order':'desc','per_page':per_query,'page':page})
   if payload is None:break
   for repo in payload.get('items',[]):
    if not repo.get('private') and not repo.get('archived') and not repo.get('fork'):
     key=repo['full_name'].lower()
     seen.setdefault(key,repo)
     buckets[qi].append(key)
    time.sleep(delay)
  # Keep searching each thematic query rather than filling only from first query.
 chosen=[]
 for name in SEEDS:
  if name.lower() in seen and name.lower() not in chosen:chosen.append(name.lower())
 for j in range(max((len(b) for b in buckets),default=0)):
  for bucket in buckets:
   if j<len(bucket) and bucket[j] not in chosen:chosen.append(bucket[j])
 chosen=chosen[:max_repos]
 rows=[]; date=datetime.now(timezone.utc).isoformat()
 for key in chosen:
  repo=seen[key]
  name=repo['full_name'];md,status=raw_readme(name,repo.get('default_branch','main'),token)
  desc=(repo.get('description') or '').strip();readme=plain(md)[:12000]
  source=f'https://github.com/{name}/blob/{repo.get("default_branch","main")}/README.md'
  category,scores=rule_label(desc+' '+readme[:2500])
  verdict='included' if readme and len(desc+' '+readme)>100 and category!='REVIEW' else 'review_or_excluded'
  logs.append({'repository':name,'source_url':source,'http_status':status,'status':verdict,'category_rule':category,'crawled_at':date})
  if not md or len(desc+' '+readme)<100:continue
  rows.append({'name':repo['name'],'full_name':name,'description':desc,'readme_excerpt':readme[:2500],
   'text':(desc+' '+readme[:2500]).strip(),'category':category,'stars':repo.get('stargazers_count',0),
   'language':repo.get('language') or '', 'github_url':repo.get('html_url') or '',
   'website_url':repo.get('homepage') or '', 'source_url':source,'crawled_at':date,
   'rule_scores':json.dumps(scores)})
  time.sleep(delay)
 raw=pd.DataFrame(rows)
 if raw.empty:raise RuntimeError('No README was fetched. Check internet access / GitHub API rate limit.')
 raw=raw.drop_duplicates('full_name').reset_index(drop=True)
 raw.to_csv(DATA/'agents_raw.csv',index=False)
 pd.DataFrame(logs).to_csv(DATA/'crawling_log.csv',index=False)
 raw[['full_name','category','description','source_url','rule_scores']].to_csv(DATA/'label_audit.csv',index=False)
 clean=raw[raw.category.isin(CATEGORIES)].copy()
 clean=clean[clean.text.str.len()>=100].reset_index(drop=True)
 clean.to_csv(DATA/'agents_clean.csv',index=False)
 print('Raw:',len(raw),'Clean:',len(clean),'Distribution:',clean.category.value_counts().to_dict())
 print('IMPORTANT: review data/label_audit.csv: labels are automated weak labels, not verified ground truth.')
 return clean

def metrics_for(model,X,y):
 pred=model.predict(X)
 p,r,f,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
 return {'accuracy':round(float(accuracy_score(y,pred)),4),'precision_macro':round(float(p),4),
 'recall_macro':round(float(r),4),'f1_macro':round(float(f),4),
 'confusion_matrix':confusion_matrix(y,pred,labels=CATEGORIES).tolist(),
 'classification_report':classification_report(y,pred,labels=CATEGORIES,zero_division=0,output_dict=True)}

def train():
 path=DATA/'agents_clean.csv'
 if not path.exists():raise FileNotFoundError('Dataset missing: run crawl first.')
 df=pd.read_csv(path).fillna('').drop_duplicates('full_name')
 df=df[df.category.isin(CATEGORIES) & (df.text.str.len()>=100)].reset_index(drop=True)
 counts=df.category.value_counts()
 if len(df)<24 or any(counts.get(c,0)<6 for c in CATEGORIES):
  raise ValueError(f'Not enough valid data for fair training. Minimum 24 rows and 6 per category. Found {counts.to_dict()}. Increase crawl coverage and manually review labels; do not fabricate records.')
 # Split before fitting TF-IDF => no leakage.
 ix_train,ix_test=train_test_split(df.index,test_size=.25,random_state=42,stratify=df.category)
 X_train=df.loc[ix_train,'text'];y_train=df.loc[ix_train,'category']
 X_test=df.loc[ix_test,'text'];y_test=df.loc[ix_test,'category']
 common=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=4000,min_df=1,sublinear_tf=True)
 nb=Pipeline([('tfidf',common),('model',MultinomialNB(alpha=.8))]);nb.fit(X_train,y_train)
 dt=Pipeline([('tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=4000,sublinear_tf=True)),
              ('model',DecisionTreeClassifier(max_depth=8,min_samples_leaf=2,class_weight='balanced',random_state=42))]);dt.fit(X_train,y_train)
 # KNN catalog only fits on texts available at deployment; does not claim test-set classification metrics.
 knn_vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=5000,sublinear_tf=True)
 matrix=knn_vectorizer.fit_transform(df.text)
 knn=NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=min(5,len(df))).fit(matrix)
 for label,model in [('naive_bayes',nb),('decision_tree',dt)]:joblib.dump(model,MODELS/(label+'.joblib'))
 joblib.dump({'vectorizer':knn_vectorizer,'model':knn,'catalog':df.to_dict('records')},MODELS/'knn.joblib')
 report={'dataset_size':len(df),'train_size':len(ix_train),'test_size':len(ix_test),
   'class_distribution':{k:int(v) for k,v in counts.items()},
   'classes':CATEGORIES,'label_quality':'weak labels from documented keyword rules; manual audit required',
   'naive_bayes':metrics_for(nb,X_test,y_test),'decision_tree':metrics_for(dt,X_test,y_test),
   'knn_evaluation':'No relevance judgments supplied; Precision@5 not fabricated.',
   'evaluated_at':datetime.now(timezone.utc).isoformat()}
 (MODELS/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 print('NB accuracy:',report['naive_bayes']['accuracy'],'DT accuracy:',report['decision_tree']['accuracy'])
 return report

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('action',choices=['crawl','train','all']);p.add_argument('--per-query',type=int,default=20);p.add_argument('--max-repos',type=int,default=85)
 a=p.parse_args()
 if a.action in ('crawl','all'):crawl(token=os.getenv('GITHUB_TOKEN','') or None,per_query=a.per_query,max_repos=a.max_repos)
 if a.action in ('train','all'):train()

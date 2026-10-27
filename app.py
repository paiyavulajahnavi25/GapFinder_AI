import os, re, json, uuid, tempfile
from pathlib import Path
from datetime import datetime
from urllib.parse import quote_plus
from flask import Flask, request, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parent
STORAGE_ROOT=Path(tempfile.gettempdir())/'gapfinder_ai' if os.getenv('VERCEL') == '1' else ROOT
UPLOADS=STORAGE_ROOT/'uploads'; DATA=STORAGE_ROOT/'data'
UPLOADS.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
USERS=DATA/'users.json'; STATE=DATA/'state.json'
app=Flask(__name__,static_folder=None); app.secret_key=os.getenv('GAPFINDER_SECRET','local-demo-change-this-secret')

def readj(path, default):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except Exception: return default
def writej(path, data): path.write_text(json.dumps(data,indent=2),encoding='utf-8')
def state(): return readj(STATE,{'materials':[],'history':[],'wrong_topics':[],'stats':{'correct':0,'attempts':0}})
def extract(path):
    ext=path.suffix.lower()
    if ext=='.txt': return path.read_text(encoding='utf-8',errors='ignore')
    if ext=='.pdf': return '\n'.join(p.extract_text() or '' for p in PdfReader(str(path)).pages)
    if ext=='.docx': return '\n'.join(p.text for p in Document(str(path)).paragraphs)
    return ''
def chunks(text,size=850,overlap=120):
    text=re.sub(r'\s+',' ',text).strip(); out=[]; start=0
    while start<len(text):
        end=min(len(text),start+size)
        if end<len(text):
            boundary=max(text.rfind('. ',start+size//2,end),text.rfind('? ',start+size//2,end),text.rfind('! ',start+size//2,end))
            if boundary>start: end=boundary+1
        if text[start:end].strip(): out.append(text[start:end].strip())
        if end>=len(text): break
        start=max(start+1,end-overlap)
    return out
def retrieve(query,k=4):
    passages=[]
    for m in state()['materials']:
        for i,t in enumerate(m['chunks']): passages.append({'text':t,'source':m['filename'],'chunk':i+1})
    if not passages:return []
    try:
        vec=TfidfVectorizer(stop_words='english',ngram_range=(1,2)); mat=vec.fit_transform([p['text'] for p in passages]+[query])
        scores=cosine_similarity(mat[-1],mat[:-1]).ravel(); ids=scores.argsort()[::-1][:k]
        return [{**passages[i],'score':round(float(scores[i]),3)} for i in ids if scores[i]>0]
    except ValueError:return []
def answer_with_context(question,ctx):
    key=os.getenv('OPENAI_API_KEY')
    if key and ctx:
        try:
            from openai import OpenAI
            context='\n\n'.join(f"[Source: {c['source']}, passage {c['chunk']}]\n{c['text']}" for c in ctx)
            r=OpenAI(api_key=key).chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'),temperature=.2,messages=[
              {'role':'system','content':'You are a helpful student tutor. Answer only from the provided study passages. If they do not answer the question, say so. Cite source filenames.'},
              {'role':'user','content':f'Study passages:\n{context}\n\nQuestion: {question}'}])
            return r.choices[0].message.content.strip(),'OpenAI + local RAG'
        except Exception: pass
    if ctx:return '\n\n'.join(f"From {c['source']} (passage {c['chunk']}):\n{c['text']}" for c in ctx[:3]),'Local RAG retrieval'
    return 'I could not find a matching passage. Upload relevant notes and try again.','No matching source'
@app.get('/')
def home(): return send_from_directory(ROOT/'public','index3.html')
@app.get('/login')
def login_page(): return send_from_directory(ROOT/'public','login.html')
@app.get('/settings')
def settings_page(): return send_from_directory(ROOT/'public','settings.html')
@app.get('/api/session')
def session_info(): return jsonify(logged_in=bool(session.get('email')),name=session.get('name','Learner'))
@app.post('/api/signup')
def signup():
    d=request.get_json(force=True); name=d.get('name','').strip(); email=d.get('email','').strip().lower(); password=d.get('password','')
    if not name or not email or len(password)<6:return jsonify(error='Enter your name, valid email, and password with at least 6 characters.'),400
    users=readj(USERS,{})
    if email in users:return jsonify(error='Account already exists. Please sign in.'),409
    users[email]={'name':name,'hash':generate_password_hash(password)};writej(USERS,users);session['email']=email;session['name']=name
    return jsonify(ok=True)

@app.post('/api/login')
def login():
    d=request.get_json(force=True); email=d.get('email','').strip().lower(); user=readj(USERS,{}).get(email)
    if not user or not check_password_hash(user['hash'],d.get('password','')):return jsonify(error='Incorrect email or password.'),401
    session['email']=email;session['name']=user['name'];return jsonify(ok=True)
@app.post('/api/logout')
def logout(): session.clear();return jsonify(ok=True)
@app.get('/api/dashboard')
def dashboard():
    s=state(); st=s['stats']; accuracy=round(100*st['correct']/st['attempts']) if st['attempts'] else 0
    return jsonify(name=session.get('name','Learner'),materials=[{k:m[k] for k in ('id','filename','uploaded_at','chunks_count')} for m in s['materials']],stats=st,accuracy=accuracy,wrong_topics=s['wrong_topics'][-20:],history=s['history'][-10:])
@app.post('/api/upload')
def upload():
    f=request.files.get('file')
    if not f:return jsonify(error='Choose a file.'),400
    name=secure_filename(f.filename or ''); ext=Path(name).suffix.lower()
    if ext not in ('.pdf','.txt','.docx'):return jsonify(error='Upload PDF, TXT, or DOCX files only.'),400
    stored=uuid.uuid4().hex+'_'+name; path=UPLOADS/stored;f.save(path)
    try: text=extract(path)
    except Exception as e:path.unlink(missing_ok=True);return jsonify(error='Could not read file: '+str(e)),400
    parts=chunks(text)
    if not parts:path.unlink(missing_ok=True);return jsonify(error='No readable text found. Scanned PDFs need OCR.'),400
    s=state();s['materials'].append({'id':uuid.uuid4().hex[:10],'filename':name,'stored':stored,'uploaded_at':datetime.now().strftime('%Y-%m-%d %H:%M'),'chunks':parts,'chunks_count':len(parts)});writej(STATE,s)
    return jsonify(ok=True,message=f'{name} processed into {len(parts)} searchable passages.')
@app.delete('/api/materials/<mid>')
def delete_material(mid):
    s=state(); item=next((m for m in s['materials'] if m['id']==mid),None)
    if not item:return jsonify(error='Material not found.'),404
    (UPLOADS/item['stored']).unlink(missing_ok=True);s['materials']=[m for m in s['materials'] if m['id']!=mid];writej(STATE,s);return jsonify(ok=True)
@app.post('/api/ask')
def ask():
    q=request.get_json(force=True).get('question','').strip()
    if not q:return jsonify(error='Type a question first.'),400
    ctx=retrieve(q); ans,mode=answer_with_context(q,ctx);s=state();s['history'].append({'question':q,'answer':ans,'at':datetime.now().strftime('%Y-%m-%d %H:%M')});writej(STATE,s)
    return jsonify(answer=ans,mode=mode,sources=ctx)
@app.post('/api/quiz')
def quiz():
    topic=request.get_json(force=True).get('topic','').strip();ctx=retrieve(topic,5)
    if not ctx:return jsonify(error='No matching notes found. Upload material or try a topic from your notes.'),404
    key=os.getenv('OPENAI_API_KEY')
    if key:
        try:
            from openai import OpenAI
            context='\n'.join(c['text'] for c in ctx)
            r=OpenAI(api_key=key).chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'),temperature=.2,response_format={'type':'json_object'},messages=[
              {'role':'system','content':'Create one multiple choice question based only on the notes. Return JSON keys question, options (4 strings), answer_index (0-3), explanation.'},
              {'role':'user','content':f'Topic: {topic}\nNotes:\n{context}'}])
            result=json.loads(r.choices[0].message.content);result['sources']=list(dict.fromkeys(c['source'] for c in ctx));return jsonify(result)
        except Exception: pass
    passage=ctx[0]['text'];opts=[passage[:220]+('…' if len(passage)>220 else ''),'The topic is unrelated to the uploaded notes.','The notes contain no information about this topic.','This option is not supported by the retrieved passage.']
    return jsonify(question=f'Which option is supported by your notes about “{topic}”?',options=opts,answer_index=0,explanation=f'The first option is taken from {ctx[0]["source"]}.',sources=list(dict.fromkeys(c['source'] for c in ctx)))
@app.post('/api/quiz-result')
def quiz_result():
    d=request.get_json(force=True);s=state();s['stats']['attempts']+=1
    if d.get('correct'):s['stats']['correct']+=1
    else:s['wrong_topics'].append({'topic':d.get('topic','General'),'at':datetime.now().strftime('%Y-%m-%d %H:%M')})
    writej(STATE,s);return jsonify(ok=True)
@app.get('/api/videos')
def videos():return jsonify(url='https://www.youtube.com/results?search_query='+quote_plus(request.args.get('topic','study skills')+' explained'))
@app.get('/<path:filename>')
def static_files(filename):
    if filename in ('style.css','app.js','login.js','settings.html'):return send_from_directory(ROOT/'public',filename)
    return jsonify(error='Not found'),404
if __name__=='__main__': app.run(debug=True,port=5001)

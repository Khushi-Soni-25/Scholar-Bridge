import os, sys, csv, asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from bson import ObjectId

load_dotenv(Path(__file__).with_name('.env'))
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import get_db
from auth import hash_password, verify_password, create_token, decode_token
from schemas import AuthRegister, AuthLogin, Profile, RecommendationRequest, SearchRequest, ExplainRequest, SaveRequest
from scholarbridge_engine.models import StudentProfile
from scholarbridge_engine.engine import ScholarBridgeEngine
from scholarbridge_engine.evidence.retriever import get_evidence_retriever
from scholarbridge_engine.orchestration.service import LangChainOrchestrationService
from scholarbridge_engine.workflow.service import ScholarBridgeWorkflowService

app = FastAPI(title='ScholarBridge API', version='2.0.0')
origins = os.getenv('FRONTEND_ORIGIN', 'http://localhost:5173').split(',')
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

engine = None
retriever = None
orchestrator = None
workflow = None

@app.on_event('startup')
async def startup():
    global engine, retriever, orchestrator, workflow
    engine = ScholarBridgeEngine.from_preferred_source(os.getenv('MONGODB_URI'), os.getenv('MONGODB_DB_NAME'))
    try:
        retriever = get_evidence_retriever()
    except Exception:
        retriever = None
    try:
        orchestrator = LangChainOrchestrationService(engine=engine, retriever=retriever) if retriever else None
    except Exception:
        orchestrator = None
    try:
        workflow = ScholarBridgeWorkflowService()
    except Exception:
        workflow = None

async def current_user(authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(401, 'Authentication required')
    uid = decode_token(authorization.split(' ',1)[1])
    if not uid:
        raise HTTPException(401, 'Invalid or expired token')
    db = get_db()
    if db is None:
        raise HTTPException(503, 'MongoDB Atlas is not configured')
    user = await db.users.find_one({'_id': ObjectId(uid)})
    if not user:
        raise HTTPException(401, 'User not found')
    user['id'] = str(user.pop('_id'))
    user.pop('password_hash', None)
    return user

def profile_engine(p: Profile):
    return StudentProfile(**p.model_dump())

def serialize_rec(item):
    s, m = item.scholarship, item.match_result
    return {
        'rank': item.rank, 'scholarship_id': s.scholarship_id, 'scholarship_name': s.scholarship_name,
        'scheme_code': s.scheme_code, 'eligibility_status': m.status.value,
        'final_score': round(item.final_score,2), 'benefit_score': round(item.benefit_score,2),
        'confidence_score': round(item.confidence_score,2), 'urgency_score': round(item.urgency_score,2),
        'specificity_score': round(item.specificity_score,2), 'explanation': m.explanation or '',
        'portal': s.application_portal or '', 'deadline_status': s.deadline_status or '',
        'deadline': s.application_end_date or '', 'max_benefit_inr': s.max_benefit_amount_inr or 0,
        'provider': s.provider_name or '', 'official_website': s.official_website or '',
        'official_source_url': s.official_source_url or '', 'benefit_description': s.benefit_description or '',
        'income_limit': s.annual_income_limit_inr or '', 'education_level': s.education_level_normalized or '',
        'category': s.eligible_categories or '', 'gender': s.gender or '',
        'application_start_date': s.application_start_date or ''
    }

def scholarship_dict(s):
    return {
      'scholarship_id':s.scholarship_id,'scheme_code':s.scheme_code,'scholarship_name':s.scholarship_name,
      'short_name':s.short_name,'scholarship_type':s.scholarship_type,'government_level':s.government_level,
      'provider_name':s.provider_name,'provider_type':s.provider_type,'department':s.department,
      'education_level':s.education_level_normalized,'stream':s.stream,'eligible_categories':s.eligible_categories,
      'gender':s.gender,'disability_eligible':s.disability_eligible,'special_beneficiary_group':s.special_beneficiary_group,
      'academic_criteria':s.academic_criteria,'annual_income_limit':s.annual_income_limit_inr,
      'benefit_type':s.benefit_type,'benefit_description':s.benefit_description,'max_benefit_inr':s.max_benefit_amount_inr,
      'application_portal':s.application_portal,'official_website':s.official_website,'application_start_date':s.application_start_date,
      'application_end_date':s.application_end_date,'deadline_status':s.deadline_status,'gujarat_applicability':s.gujarat_applicability,
      'official_source_url':s.official_source_url,'verification_status':s.verification_status,'status':s.status,
    }

@app.get('/api/health')
async def health():
    db = get_db(); mongo_ok = False
    if db is not None:
        try: await db.command('ping'); mongo_ok=True
        except Exception: pass
    return {'status':'ok','mongo_connected':mongo_ok,'engine':'ready' if engine else 'starting','rag':'ready' if retriever else 'unavailable','langchain':bool(orchestrator),'langgraph':bool(workflow)}

@app.post('/api/auth/register')
async def register(req: AuthRegister):
    db=get_db()
    if db is None: raise HTTPException(503,'Set MONGODB_URI in backend/.env first')
    if await db.users.find_one({'email':req.email.lower()}): raise HTTPException(409,'An account with this email already exists')
    doc={'name':req.name.strip(),'email':req.email.lower(),'password_hash':hash_password(req.password),'created_at':__import__('datetime').datetime.utcnow()}
    r=await db.users.insert_one(doc)
    return {'token':create_token(str(r.inserted_id)),'user':{'id':str(r.inserted_id),'name':doc['name'],'email':doc['email']}}

@app.post('/api/auth/login')
async def login(req: AuthLogin):
    db=get_db()
    if db is None: raise HTTPException(503,'Set MONGODB_URI in backend/.env first')
    user=await db.users.find_one({'email':req.email.lower()})
    if not user or not verify_password(req.password,user['password_hash']): raise HTTPException(401,'Invalid email or password')
    return {'token':create_token(str(user['_id'])),'user':{'id':str(user['_id']),'name':user['name'],'email':user['email']}}

@app.get('/api/auth/me')
async def me(user=Depends(current_user)): return user

@app.get('/api/profile')
async def get_profile(user=Depends(current_user)):
    db=get_db(); p=await db.profiles.find_one({'user_id':user['id']});
    if p: p.pop('_id',None); p.pop('user_id',None)
    return p or {}

@app.put('/api/profile')
async def save_profile(profile: Profile, user=Depends(current_user)):
    db=get_db(); data=profile.model_dump(); data['user_id']=user['id']; data['updated_at']=__import__('datetime').datetime.utcnow()
    await db.profiles.update_one({'user_id':user['id']},{'$set':data},upsert=True)
    return {'saved':True,'profile':profile}

@app.get('/api/scholarships')
async def scholarships(q:str='', category:str='', education:str='', limit:int=50):
    items=engine.kb.scholarships
    q=q.lower().strip(); category=category.lower().strip(); education=education.lower().strip()
    out=[]
    for s in items:
        hay=f'{s.scholarship_name} {s.short_name} {s.scheme_code} {s.provider_name}'.lower()
        if q and q not in hay: continue
        if category and category not in s.eligible_categories.lower(): continue
        if education and education not in s.education_level_normalized.lower(): continue
        out.append(scholarship_dict(s))
        if len(out)>=min(limit,200): break
    return {'items':out,'total':len(out)}

@app.get('/api/scholarships/{sid}')
async def scholarship_detail(sid:str):
    s=engine.kb.scholarships_by_id.get(sid)
    if not s:
        for x in engine.kb.scholarships:
            if x.scheme_code.lower()==sid.lower(): s=x; break
    if not s: raise HTTPException(404,'Scholarship not found')
    d=scholarship_dict(s)
    d['benefits']=[getattr(x,'__dict__',{}) for x in engine.kb.benefits_by_scholarship.get(s.scholarship_id,[])]
    d['documents']=[getattr(x,'__dict__',{}) for x in engine.kb.docs_by_scholarship.get(s.scholarship_id,[])]
    d['rules']=[getattr(x,'__dict__',{}) for x in engine.kb.rules_by_scholarship.get(s.scholarship_id,[])]
    return d

@app.post('/api/recommend')
async def recommend(req: RecommendationRequest, user=Depends(current_user)):
    try:
        output=engine.recommend(profile_engine(req.profile), top_n_eligible=12, top_n_conditional=6)
        recs=[serialize_rec(x) for x in output.ranked_recommendations[:12]]
        cond=[serialize_rec(x) for x in output.conditional_recommendations[:6]]
        if get_db() is not None:
            await get_db().searches.insert_one({'user_id':user['id'],'profile':req.profile.model_dump(),'result_count':len(recs),'created_at':__import__('datetime').datetime.utcnow()})
        return {'total_evaluated':output.total_scholarships_evaluated,'eligible_count':output.eligible_count,'conditional_count':output.conditional_count,'ineligible_count':output.ineligible_count,'recommendations':recs,'conditional_recommendations':cond}
    except Exception as e: raise HTTPException(422,str(e))

@app.post('/api/rag/search')
async def rag_search(req:SearchRequest, user=Depends(current_user)):
    if not retriever: raise HTTPException(503,'RAG index is not available. Check Qdrant/embedding setup.')
    pack=retriever.retrieve(query=req.query, scheme_id=req.scheme_id, top_k=req.top_k)
    return {'query':pack.query,'generated_at':pack.generated_at,'evidence':[{'citation_id':x.citation.citation_id,'scheme_id':x.scheme_id,'document_title':x.citation.document_title,'section':x.section,'subsection':x.subsection,'page':x.page,'text':x.text,'retrieval_score':x.retrieval_score,'rank':i+1,'retrieval_source':x.retrieval_source,'source_url':x.citation.source_url or ''} for i,x in enumerate(pack.evidence)]}

@app.post('/api/explain')
async def explain(req:ExplainRequest, user=Depends(current_user)):
    if not orchestrator: raise HTTPException(503,'LangChain/RAG service is unavailable')
    result=orchestrator.explain(profile_engine(req.profile),req.scholarship_id_or_code,req.user_query)
    return result.model_dump() if hasattr(result,'model_dump') else result

@app.post('/api/workflow')
async def run_workflow(payload:dict, user=Depends(current_user)):
    if not workflow: raise HTTPException(503,'LangGraph workflow is unavailable')
    return workflow.run(user_query=payload.get('user_query'), profile=payload.get('profile'), selected_scholarships=payload.get('selected_scholarships'), thread_id=payload.get('thread_id'), intent_override=payload.get('intent_override'))

@app.get('/api/saved')
async def saved(user=Depends(current_user)):
    db=get_db(); rows=[]
    async for r in db.saved_scholarships.find({'user_id':user['id']}): rows.append(r['scholarship_id'])
    return {'items':rows}

@app.post('/api/saved')
async def save(req:SaveRequest,user=Depends(current_user)):
    db=get_db(); await db.saved_scholarships.update_one({'user_id':user['id'],'scholarship_id':req.scholarship_id},{'$set':{'user_id':user['id'],'scholarship_id':req.scholarship_id,'saved_at':__import__('datetime').datetime.utcnow()}},upsert=True); return {'saved':True}

@app.delete('/api/saved/{sid}')
async def unsave(sid:str,user=Depends(current_user)):
    db=get_db(); await db.saved_scholarships.delete_one({'user_id':user['id'],'scholarship_id':sid}); return {'saved':False}

@app.get('/api/dashboard')
async def dashboard(user=Depends(current_user)):
    db=get_db(); saved_count=await db.saved_scholarships.count_documents({'user_id':user['id']}); searches=await db.searches.count_documents({'user_id':user['id']})
    profile=await db.profiles.find_one({'user_id':user['id']});
    return {'saved_count':saved_count,'searches':searches,'profile_complete':bool(profile and profile.get('course_name'))}

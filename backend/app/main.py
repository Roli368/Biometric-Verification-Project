from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os, time
import base64
from .services import session as session_svc
from .services import alignment, quality, liveness, verification
from .config import UPLOAD_FOLDER, SESSION_TIMEOUT, MIN_BUFFER, TRUST_THRESHOLD, MATCH_THRESHOLD, SWAP_THRESHOLD
from .utils.io import imdecode_bytes
import cv2

app = FastAPI(title='Face Verification API')

# Serve uploaded files for development
UPLOAD_DIR = UPLOAD_FOLDER
if not UPLOAD_DIR:
    UPLOAD_DIR = '.'
app.mount('/uploads', StaticFiles(directory=UPLOAD_DIR), name='uploads')

# Allow CORS from local frontend during development
app.add_middleware(
    CORSMiddleware,
    # Allow common local dev ports
    allow_origins=["http://localhost:3000", "http://localhost:3002", "http://127.0.0.1:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "message": "Face Verification API", "docs": "http://localhost:8000/docs"}


@app.get('/health')
def health():
    return {'status': 'ok', 'time': time.time()}

@app.get('/api/health')
def api_health():
    return health()

@app.post('/start-session')
def start_session():
    sess = session_svc.create_session()
    return {'status':'OK', 'session_id': sess['session_id']}

@app.post('/api/start-session')
def api_start_session():
    return start_session()


@app.post('/upload-document')
async def upload_document(file: UploadFile = File(...), session_id: str = Form(...)):
    try:
        sess = session_svc.get_session(session_id)
        if not sess:
            return JSONResponse({'status':'FAILED','message':'Invalid session'}, status_code=400)
        data = await file.read()
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        doc_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_doc.jpg")
        with open(doc_path, 'wb') as f:
            f.write(data)
        # extract face
        out_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_id_face.jpg")
        res = alignment.extract_face_from_id(doc_path, out_path)
        if res is None:
            return {'status':'FAILED','message':'No face detected'}
        # compute id embedding (dummy)
        img = imdecode_bytes(data)
        id_face = cv2.imread(out_path)
        emb = verification.infer_embedding(id_face)
        session_svc.set_session(session_id, {
            'id_embedding': emb, 
            'doc_face_path': out_path,
            'liveness_passed': False,  # Reset state
            'locked_live_embedding': None,
            'attempts': 0
        })
        # Return HTTP URL for the image, not file path
        preview_url = f"/uploads/{session_id}_id_face.jpg"
        return {'status':'DOCUMENT_READY','message':'ID face extracted', 'preview': preview_url}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'status':'ERROR','message':f'Internal Error: {str(e)}'}, status_code=500)

@app.post('/api/upload-document')
async def api_upload_document(file: UploadFile = File(...), session_id: str = Form(...)):
    return await upload_document(file, session_id)


@app.post('/process-frame')
async def process_frame(frame: UploadFile = File(...), session_id: str = Form(...)):
    sess = session_svc.get_session(session_id)
    if not sess:
        return JSONResponse({'status':'FAILED','message':'Invalid session'}, status_code=400)
    try:
        # expiry
        if time.time() - sess['start_ts'] > SESSION_TIMEOUT:
            return {'status':'EXPIRED','message':'Session expired'}
        data = await frame.read()
        img = imdecode_bytes(data)
        if img is None:
            return JSONResponse({'status':'FAILED','message':'Invalid frame'}, status_code=400)
        # quality gate
        ok, qres = quality.pre_check_quality(img)
        if not ok:
            # pause but keep liveness buffers
            session_svc.set_session(session_id, {'paused': True})
            return {'status':'PAUSED','message': qres.get('message','Quality issue'), 'reason': qres.get('reason'), 'metrics': qres.get('metrics',{})}
        else:
            if sess.get('paused'):
                session_svc.set_session(session_id, {'paused': False})
        # ensure liveness instance
        if not sess.get('liveness_instance'):
            lv = liveness.UltimateLiveness10()
            session_svc.set_session(session_id, {'liveness_instance': lv})
            sess = session_svc.get_session(session_id)
        lv = sess.get('liveness_instance')
        # if not liveness passed, call verify
        if not sess.get('liveness_passed'):
            result = lv.verify(img)
            status = result.get('status')
            trust = result.get('trust', 0.0)
            metrics = result.get('metrics', {})
            print(f"Liveness result: {status}, trust: {trust}") # Debug
            if status == 'LIVE HUMAN' and trust >= TRUST_THRESHOLD:
                # lock live embedding AND image
                print("Liveness passed, inferring lock embedding") # Debug
                emb = verification.infer_embedding(img)

                # Encode image to base64 for frontend display
                _, buffer = cv2.imencode('.jpg', img)
                b64_image = base64.b64encode(buffer).decode('utf-8')

                session_svc.set_session(session_id, {
                    'liveness_passed': True, 
                    'locked_live_embedding': emb,
                    'locked_live_image': b64_image
                })
                return {'status':'PROCESSING','message':'Liveness passed, checking identity...','metrics': metrics, 'trust': trust}
            else:
                return {'status':'PROCESSING','message':'Checking liveness...','metrics': metrics, 'trust': trust}
        # After liveness passed
        # After liveness passed
        if sess.get('liveness_passed'):
            # SINGLE SHOT VERIFICATION
            # Use the frame that passed liveness (locked_live_embedding)
            locked = sess.get('locked_live_embedding')
            
            # compare with ID embedding
            id_emb = sess.get('id_embedding')
            if id_emb is None:
                return {'status':'FAILED','message':'ID not provided'}
                
            print("Verifying locked face against ID...")
            sim = verification.cosine_sim(locked, id_emb)
            print(f"Similarity: {sim}") # Debug
            
            if sim > MATCH_THRESHOLD:
                live_img_b64 = sess.get('locked_live_image')
                return {
                    'status':'COMPLETE', 
                    'message':'Verified!', 
                    'metrics':{'similarity': sim},
                    'live_image': f"data:image/jpeg;base64,{live_img_b64}"
                }
            else:

                sess['attempts'] = sess.get('attempts',0) + 1
                session_svc.set_session(session_id, {'attempts': sess['attempts']})
                if sess['attempts'] >= 2:
                    live_img_b64 = sess.get('locked_live_image')
                    return {
                        'status': 'FAILED',
                        'message': 'Face mismatch',
                        'attempts': sess['attempts'],
                        'live_image': f"data:image/jpeg;base64,{live_img_b64}" if live_img_b64 else None
                    }
                return {'status':'PROCESSING','message':'Face not matching, try again','metrics':{'similarity': sim}, 'attempts': sess['attempts']}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'status':'ERROR','message':f'Internal Error: {str(e)}'}, status_code=500)

@app.post('/api/process-frame')
async def api_process_frame(frame: UploadFile = File(...), session_id: str = Form(...)):
    try:
        return await process_frame(frame, session_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'status':'ERROR','message':f'Internal Error: {str(e)}'}, status_code=500)
@app.get('/status/{session_id}')
def get_status(session_id: str):
    try:
        sess = session_svc.get_session(session_id)
        if not sess:
            return {'status': 'UNKNOWN', 'message': 'Session not found'}
        return {'status': sess.get('status', 'PROCESSING'), 'message': 'Session active'}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'ERROR', 'message': str(e)}

@app.post('/cleanup-session')
def cleanup_session(session_id: str = Form(...)):
    """Delete uploaded images and clean up session data after analysis."""
    try:
        sess = session_svc.get_session(session_id)
        if not sess:
            return {'status': 'OK', 'message': 'Session already expired'}
        
        # Delete uploaded document image
        doc_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_doc.jpg")
        if os.path.exists(doc_path):
            os.remove(doc_path)
            
        # Delete extracted face image
        face_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_id_face.jpg")
        if os.path.exists(face_path):
            os.remove(face_path)
            
        # Clear session data (optional - could also delete session entirely)
        session_svc.set_session(session_id, {
            'id_embedding': None,
            'doc_face_path': None,
            'liveness_passed': False,
            'locked_live_embedding': None,
            'locked_live_image': None,
            'attempts': 0
        })
        
        return {'status': 'OK', 'message': 'Session cleaned up successfully'}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'ERROR', 'message': str(e)}

@app.post('/api/cleanup-session')
def api_cleanup_session(session_id: str = Form(...)):
    return cleanup_session(session_id)
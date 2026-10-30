import re
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title='Neptune Minigame Solver')

# Terminal candidate text is frequently wrapped over the next 12-pixel memory row.
# OCR the two columns separately, then use the address increments to join fragments.
def recognize(img):
    h,w=img.shape[:2]
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # White/near-white characters (candidate words) and brighter cyan-green highlight.
    white=cv2.inRange(hsv,np.array([0,0,105]),np.array([179,145,255]))
    # Green game text (use only as fallback; address text is green).
    green=cv2.inRange(hsv,np.array([35,40,45]),np.array([100,255,255]))
    results=[]
    for maskname,mask in [('white',white),('green',green)]:
        for scale in (2,3):
            enlarged=cv2.resize(mask,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
            for psm in (6,11):
                d=pytesseract.image_to_data(enlarged,config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',output_type=pytesseract.Output.DICT)
                for i,raw in enumerate(d['text']):
                    word=re.sub('[^A-Za-z]','',raw).upper()
                    if len(word)<2 or len(word)>12: continue
                    try: conf=float(d['conf'][i])
                    except: conf=-1
                    if conf<5: continue
                    x,y,ww,hh=[int(d[k][i])/scale for k in ('left','top','width','height')]
                    results.append(dict(text=word,x=round(x),y=round(y),width=round(ww),height=round(hh),confidence=round(conf),pass_name=maskname))
    # Merge overlapping recognitions, preferring longer / higher confidence tokens.
    results.sort(key=lambda t:(-len(t['text']),-t['confidence']))
    unique=[]
    for t in results:
        if any(abs(t['x']-a['x'])<max(10,t['width']*.45) and abs(t['y']-a['y'])<max(9,t['height']*.8) for a in unique): continue
        unique.append(t)
    unique.sort(key=lambda t:(t['y'],t['x']))
    return unique

@app.post('/api/ocr')
async def ocr(file: UploadFile=File(...)):
    data=await file.read()
    if len(data)>12_000_000: raise HTTPException(413,'Image too large')
    img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400,'Invalid image')
    if img.shape[0]*img.shape[1]>12_000_000: raise HTTPException(413,'Image dimensions too large')
    return {'tokens':recognize(img),'note':'Review and correct candidate words; split-line words may need manual joining.'}

app.mount('/', StaticFiles(directory='.', html=True), name='site')

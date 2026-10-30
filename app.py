"""Neptune Solver: screenshot OCR, word matching and static web app."""
import re
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent
WORDS = set(re.findall(r'^[A-Z]{4,10}$', (ROOT/'game_words.txt').read_text(), re.M))
app = FastAPI(title='Neptune Solver')

@app.get('/')
def index():
    return FileResponse(ROOT/'index.html')

@app.get('/api/health')
def health():
    return {'ok': True, 'dictionary_size': len(WORDS), 'ocr': 'tesseract'}

def text_rows(mask, x_offset, psm):
    scaled = cv2.resize(mask, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(scaled, config=f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', output_type=pytesseract.Output.DICT)
    out = []
    for i, raw in enumerate(data['text']):
        word = re.sub('[^A-Z]', '', raw.upper())
        if not 2 <= len(word) <= 12:
            continue
        conf = float(data['conf'][i])
        if conf < 5:
            continue
        out.append({'text': word, 'x': x_offset+int(data['left'][i]/3), 'y': int(data['top'][i]/3), 'confidence': conf})
    return out

def recognize(img, word_length):
    lengths = {word_length} if word_length else set(range(4, 11))
    h,w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0,0,100), (179,135,255))
    green = cv2.inRange(hsv, (35,45,45), (105,255,255))
    tokens = []
    # Process each column independently so OCR does not merge left and right rows.
    for x0,x1 in ((0,w//2),(w//2,w)):
        for mask in (white, cv2.bitwise_or(white, green)):
            for psm in (6,11):
                tokens.extend(text_rows(mask[:,x0:x1], x0, psm))
    # Read short terminal rows separately: full-column OCR tends to return no
    # useful tokens when punctuation surrounds each candidate word.
    for x0,x1 in ((0,w//2),(w//2,w)):
        region=white[:,x0:x1]
        density=np.count_nonzero(region,axis=1)
        active=np.flatnonzero(density>=3)
        bands=[]
        for y in active:
            if not bands or y-bands[-1][-1]>7: bands.append([int(y)])
            else: bands[-1].append(int(y))
        for band in bands:
            y0,y1=band[0],band[-1]+1
            if not 7<=y1-y0<=34: continue
            # Remove the left-side hex address, keeping the terminal content.
            crop=region[max(0,y0-3):min(h,y1+4),int((x1-x0)*.26):]
            for t in text_rows(crop,x0+int((x1-x0)*.26),7):
                t['y']+=max(0,y0-3)
                tokens.append(t)
    # A punctuation-preserving pass is essential for Fallout-style terminal
    # screenshots: forcing a letters-only whitelist can erase entire words.
    for x0,x1 in ((0,w//2),(w//2,w)):
        crop=img[:,x0:x1]
        enlarged=cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        for psm in (6,11):
            data=pytesseract.image_to_data(enlarged,config=f'--oem 3 --psm {psm}',output_type=pytesseract.Output.DICT)
            for i,raw in enumerate(data['text']):
                conf=float(data['conf'][i])
                if conf<0: continue
                # Keep punctuation as boundaries, but also extract embedded
                # letter runs, e.g. "-DONGLE\\;" or "###CHARGE_&".
                for match in re.finditer(r'[A-Za-z]{3,}',raw):
                    word=match.group().upper()
                    tokens.append({'text':word,'x':x0+int(data['left'][i]/3),
                                   'y':int(data['top'][i]/3),'confidence':conf})
    # De-duplicate OCR passes by location and token.
    unique = {}
    for t in tokens:
        key = (t['text'],round(t['x']/12),round(t['y']/9))
        if key not in unique or unique[key]['confidence'] < t['confidence']:
            unique[key] = t
    tokens = sorted(unique.values(), key=lambda t:(t['y'],t['x']))
    candidates = {}
    def add(word, confidence, source):
        if len(word) not in lengths: return
        # Only verified dictionary entries reach the playable candidate list.
        # Unknown OCR strings remain available in the raw tokens for debugging.
        if word not in WORDS: return
        score = confidence + (12 if word in WORDS else -12)
        if word not in candidates or score > candidates[word]['confidence']:
            candidates[word] = {'word':word,'confidence':round(max(0,min(100,score)),1),'source':source}
    for t in tokens:
        s=t['text']
        if len(s) in lengths:
            if s in WORDS: add(s,t['confidence'],'dictionary exact')
            # Known vocabulary corrects one-character misreads; show as uncertain.
            if t['confidence'] >= 25:
                for word in WORDS:
                    if len(word)==len(s) and sum(a!=b for a,b in zip(word,s))==1 and t['confidence'] >= 75:
                        add(word,t['confidence']-65,'uncertain OCR correction')
        for length in lengths:
            if len(s)>length:
                for i in range(len(s)-length+1):
                    word=s[i:i+length]
                    if word in WORDS: add(word,t['confidence']-15,'embedded')

    # Combine terminal rows within each column, never across columns.
    for x0,x1 in ((0,w//2),(w//2,w)):
        crop=cv2.resize(img[:,x0:x1],None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        raw=pytesseract.image_to_string(crop,config='--oem 3 --psm 6').upper()
        lines=raw.splitlines()
        for a,b in zip(lines,lines[1:]):
            a=re.sub(r'^\\s*(?:0[XO])?[0-9A-F]{3,8}\\s+', '',a)
            b=re.sub(r'^\\s*(?:0[XO])?[0-9A-F]{3,8}\\s+', '',b)
            tail=re.search(r'([A-Z]{1,9})[^A-Z]*$',a)
            head=re.search(r'^[^A-Z]*([A-Z]{1,9})',b)
            if not tail or not head: continue
            left,right=tail.group(1),head.group(1)
            for word in WORDS:
                if len(word) not in lengths: continue
                for cut in range(1,len(word)):
                    if left.endswith(word[:cut]) and right.startswith(word[cut:]):
                        add(word,55,'line wrap')
    # Full-row OCR can recover complete known words even when the tokenizer
    # merges the terminal's surrounding punctuation into a single token.
    for x0,x1 in ((0,w//2),(w//2,w)):
        crop=cv2.resize(img[:,x0:x1],None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        raw=pytesseract.image_to_string(crop,config='--oem 3 --psm 6').upper()
        for line in raw.splitlines():
            compact=re.sub('[^A-Z]','',line)
            for word in WORDS:
                if len(word) in lengths and word in compact:
                    add(word,70,'raw OCR')
    # Do not automatically guess dictionary words from a single-character
    # similarity: it produces false positives when the screenshot contains hex.
    result=sorted((v for v in candidates.values() if v['source'] != 'uncertain OCR correction'),key=lambda t:(-t['confidence'],t['word']))
    return {'words':result,'tokens':tokens[:180],'dictionary_size':len(WORDS),
            'note':'Only dictionary matches are shown. Unknown words may be omitted; review the screenshot and add missing words to game_words.txt.'}

@app.post('/api/ocr')
async def ocr(file: UploadFile=File(...), word_length: int=0):
    if word_length != 0 and not 4<=word_length<=10: raise HTTPException(400,'Word length must be 0 (auto) or 4 to 10')
    data=await file.read()
    if len(data)>12_000_000: raise HTTPException(413,'Image too large')
    img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400,'Invalid image')
    if img.shape[0]*img.shape[1]>12_000_000: raise HTTPException(413,'Image dimensions too large')
    return recognize(img,word_length)

# Neptune Minigame Solver

A self-hosted word-likeness + ECU bit solver.

## Deploy to Railway (recommended)
1. Upload this folder to a new GitHub repository.
2. In Railway, choose **New Project → Deploy from GitHub repo** and select it.
3. Railway detects the Dockerfile. Deploy, then open **Settings → Networking → Generate Domain**.
4. Visit the generated URL. Paste screenshots with Ctrl+V and use **Scan screenshot**.

Railway includes server-side Tesseract OCR. OCR is imperfect with game terminal fonts; manually correct split words and bad detections. The app does not retain uploaded screenshots.

## Deploy to GitHub Pages (manual-only mode)
1. Publish the repository and enable Pages in **Settings → Pages → Deploy from branch**.
2. Use the `/public` folder content as the site root (copy `public/index.html` to repository root for branch deployment).
3. The word solver and bit solver work, but **Scan screenshot** requires the Railway backend `/api/ocr`. For full OCR, deploy to Railway instead.

## Local run
With Python 3.11, Tesseract OCR installed and dependencies installed from `requirements.txt`, run `uvicorn app:app --reload` and open http://localhost:8000.

## Features
- Paste, drop, upload screenshots; OCR tokens can be clicked into editable candidate list.
- Likeness solver filters by all feedback and chooses a guess that partitions remaining candidates.
- Clickable bits and editable hex, XOR/NOT/ROL/ROR shortest-path solver.

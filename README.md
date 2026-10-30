# Neptune Solver

A Railway-hosted screenshot word-likeness solver and ECU byte-operation solver.

## Railway deployment
1. Connect the GitHub repository `LCRPLBD/Solver` to a Railway service.
2. Railway should use the repository `Dockerfile` (Python 3.11 and Tesseract).
3. Generate a public domain under Railway Networking.
4. Open `/api/health` to confirm the service is running, then open `/` for the interface.

The Dockerfile listens on Railway's `PORT` environment variable. No API keys are required.

## Using the solver
- Paste a screenshot with Ctrl+V or upload/drop an image. Recognition runs automatically.
- The backend reads both columns separately, attempts to match candidate words against `game_words.txt`, and returns confidence-ranked suggestions.
- **Review and correct OCR suggestions** before using likeness feedback. The dictionary is a curated starting list, not a proven complete list.
- Enter likeness feedback to narrow possible words. The ECU panel solves two 8-bit registers with the selected XOR/NOT/rotation operations.

## API
- `GET /api/health` returns status and dictionary size.
- `POST /api/ocr?word_length=6` accepts a multipart `file` image and returns `words` (objects containing `word`, `confidence`, and `source`), `tokens`, and `dictionary_size`.
- Image upload limit: 12 MB and 12 million pixels. Images are processed in memory, not saved.

## Known limitations
OCR may miss words embedded in terminal symbol noise or wrapped between rows. One-character dictionary corrections are low confidence and may be false positives. The recognizer has not been benchmarked against all historical screenshots; do not assume the 93-word vocabulary is exhaustive.

## Local development
Install Python 3.11, Tesseract OCR, and dependencies from `requirements.txt`. Run `uvicorn app:app --reload`, then open http://localhost:8000.

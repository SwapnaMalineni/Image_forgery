Image Forgery Detector — quick deploy notes

This repo contains a Flask app that performs ELA-based forgery detection using a Keras model.

Quick summary of the deployment changes made:
- `build.sh` downloads the model into `/mnt/data/forgery_model.h5` when available.
- `start.sh` sets `TF_CPP_MIN_LOG_LEVEL=2` and `CUDA_VISIBLE_DEVICES=""` then runs `gunicorn -w 1 --bind 0.0.0.0:$PORT app:app`.
- `requirements.txt` uses `tensorflow-cpu` to avoid GPU driver probing on CPU-only hosts.
- `app.py` now includes a `GET /health` endpoint which returns JSON: `{ok, model_loaded, model_path}` and has robust model lookup logic.

Recommended Render setup (fastest, reproducible)
1. In your repository root ensure `start.sh` is executable and committed.
2. In Render dashboard, create a new Web Service (or edit your service) and set:
   - Environment: `Python 3`
   - Build Command: `./build.sh`
   - Start Command: `./start.sh`
3. Add these Environment Variables in Render (Dashboard → Environment):
   - `TF_CPP_MIN_LOG_LEVEL` = `2`
   - `CUDA_VISIBLE_DEVICES` = `""` (empty string)
4. Deploy.

Verification
- Visit `https://<your-render-url>/health` and confirm `"model_loaded": true` and `model_path` shows where the model was loaded from.
- Then POST an image to `/api/detect-forgery` (example using curl):
  curl.exe -v -F "file=@C:\path\to\image.jpg" https://<your-render-url>/api/detect-forgery

If the model is not loaded
- Check the Render build logs for "Downloading model to:" and "Model downloaded successfully to" messages.
- Check the Render server logs for the model-loading diagnostics printed by `get_model()` in `app.py` — they list tried paths and any load errors.
- If Google Drive download fails repeatedly consider uploading `forgery_model.h5` to an S3 bucket or using `gdown` with a confirmed id.

Security notes
- Keep `forgery_model.h5` out of git history. Use Render's persistent disk, S3, or a release asset.
- Keep secrets (SECRET_KEY, MAIL_PASSWORD) in Render Environment variables, not in code.

Contact
- If you want, I can add a small `render.yaml` for you or swap model download to `gdown`. Reply with which you prefer.

Quick fix - Update the user loader to handle missing tables:

```python
# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception as e:
        print(f"Database error in load_user: {e}")
        return None
```

To fix this immediately:

1. **Push the updated files** (`init_db.py` and `start.sh`) I created
2. **Clear build cache & deploy**
3. **Check logs** for database initialization messages

Alternative quick fix:
- In Render Shell, run: `python -c "from app import app, db; app.app_context().push(); db.create_all(); print('DB created')"`

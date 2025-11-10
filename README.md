# Defect Detector — Deployment Guide

This repository contains a Streamlit app (`app.py`) for defect detection using a Keras model (`mobilenetv2_finetuned_best_model.keras`). This README explains two simple deployment options: Streamlit Cloud (fast) and Docker (flexible). It also explains model-hosting considerations.

## Important note about the model file

The model file (`mobilenetv2_finetuned_best_model.keras`) can be large. If it's too large to commit to a GitHub repo, consider one of the following:

- Use Git LFS to store the model in your GitHub repository.
- Host the model on cloud storage (S3, Google Cloud Storage) and download it at runtime (set a `MODEL_URL` environment variable and update `app.py` accordingly).

If you plan to deploy to Streamlit Cloud, check the file size limits — large model files are better hosted externally.

---

## Option A — Streamlit Cloud (recommended for simple deployments)

1. Push your project to a GitHub repository (public or private).
2. Make sure the repo includes `app.py` and `requirements.txt`.
3. If the model file is too large, either add it via Git LFS or host it externally and set the `MODEL_URL` environment variable in Streamlit Cloud. The app will download the model at startup if a local copy is not present.
4. Go to https://share.streamlit.io and connect your GitHub repo.
5. Select the branch and the `app.py` file. Streamlit Cloud will install packages from `requirements.txt` automatically and launch the app.

Streamlit Cloud is quick and easy, but you'll need to ensure the model is accessible to the deployed app (via LFS or external URL).

---

## Option B — Docker (flexible; works on cloud providers / VPS)

This repository includes a `Dockerfile` you can use to build an image and run the app.

Build the image (PowerShell):

```powershell
docker build -t defect-detector:latest .
```

Run locally:

```powershell
docker run --rm -p 8501:8501 -v ${PWD}:/app defect-detector:latest
```

Notes:
- The Dockerfile uses `python:3.10-slim`. If you require Python 3.12 for TensorFlow compatibility, edit the `FROM` line.
- If your model file is not present in the repo, mount it into `/app` (or copy it during image build) so the container can load it. Example mounting a local `models` folder:

```powershell
docker run --rm -p 8501:8501 -v ${PWD}:/app -v ${PWD}\models:/app/models defect-detector:latest
```

---

## Model hosting options

- Git LFS: good for keeping model alongside code. Requires Git LFS enabled in the repo.
- Cloud storage: upload model to S3/GCS and set an environment variable `MODEL_URL` used by `app.py` to download the model on first run.
- Containerize the model: copy the model into the Docker image if size is acceptable and you control the image registry.

---

## Quick troubleshooting

- If `cv2` was missing previously, ensure `opencv-python` is in `requirements.txt` (it is already present). When building the Docker image, the package will be installed.
- If Streamlit can't find the model, check the working directory and file names. You can also set `MODEL_PATH` or `MODEL_URL` environment variables and adapt `app.py`.

---

If you'd like, I can:

- Add model-download logic to `app.py` so missing models are fetched from an URL set by `MODEL_URL`.
- Create a GitHub Actions workflow to build and push a Docker image to GitHub Container Registry.
- Help configure Streamlit Cloud deployment and show the exact step-by-step commands for your repository.

Tell me which of these you'd like me to implement next.

web: gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 1 --timeout 120

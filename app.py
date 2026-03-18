# Render / gunicorn entry point.
# Imports the Flask app object from main.py.
# Background threads and DB init are handled by gunicorn.conf.py's on_starting hook.
from main import app  # noqa: F401

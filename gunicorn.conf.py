"""Gunicorn configuration for the physiological monitoring system.

Uses a single worker with threads so the background demo/video thread
shares memory with the Flask worker (multi-process would isolate state).
The on_starting hook initialises the DB and launches the background worker
before any requests are served.
"""

import threading
import os

# Single worker + threads so the background thread is in the same process
workers = 1
threads = 4
worker_class = 'gthread'
timeout = 120
preload_app = True          # import app once in master before forking


def on_starting(server):
    """Called once in the master process before workers are forked."""
    import signal as _signal
    # Suppress default gunicorn SIGTERM re-raise so our handler runs cleanly
    from main import _init_db, _demo_loop, process_video, stop_event, logger
    from config import DEMO_MODE

    _init_db()

    target = _demo_loop if DEMO_MODE else process_video
    name   = 'DemoLoop' if DEMO_MODE else 'VideoProcessing'
    t = threading.Thread(target=target, daemon=True, name=name)
    t.start()
    logger.info(f'[gunicorn] {name} thread started (pid={os.getpid()})')

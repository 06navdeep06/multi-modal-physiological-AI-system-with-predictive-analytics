"""Render entry point.

Render's dashboard start command is hardcoded to 'python app.py'.
This file initialises the DB, starts the background worker thread,
then runs the Flask server — identical to main.py's __main__ block
but always executes regardless of __name__.
"""

import os
import sys
import signal
import threading

from main import (
    app, _init_db, _demo_loop, process_video,
    stop_event, logger,
)
from config import DEMO_MODE, FLASK_HOST, FLASK_PORT, FLASK_DEBUG


def _signal_handler(signum, _frame):
    logger.info(f'Signal {signum} – shutting down')
    stop_event.set()
    sys.exit(0)


_init_db()

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

_target = _demo_loop if DEMO_MODE else process_video
_name   = 'DemoLoop' if DEMO_MODE else 'VideoProcessing'
threading.Thread(target=_target, daemon=True, name=_name).start()
logger.info(f'{_name} thread started')

logger.info(f'Starting server on {FLASK_HOST}:{FLASK_PORT}')
app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False, threaded=True)

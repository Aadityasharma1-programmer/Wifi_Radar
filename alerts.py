"""Fall alert logging + optional webhook (Slack/Discord/ntfy-style) notification."""
import json
import os
import threading
import time
import urllib.request
from config import LOG_PATH


def log_event(ev):
    with open(LOG_PATH, "a") as f:
        f.write(f"{ev['time']},{ev['prob']:.2f},{ev['type']}\n")


def notify(ev):
    url = os.getenv("WEBHOOK_URL")
    if not url:
        return
    msg = f"Possible fall detected at {ev['time']} (confidence {ev['prob']:.0%})"

    def _post():
        try:
            req = urllib.request.Request(url, data=json.dumps({"text": msg, "content": msg}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
    threading.Thread(target=_post, daemon=True).start()


def make_event(prob):
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "prob": float(prob), "type": "fall-like movement"}

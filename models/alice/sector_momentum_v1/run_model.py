"""Example Python model that emits simple signals."""
from __future__ import annotations

import json
import sys
from datetime import datetime


def main() -> None:
    payload = json.load(sys.stdin)
    prices = payload["data"].get("prices", [])
    signals = []
    for security in prices:
        if not security.get("dates"):
            continue
        ts = security["dates"][0]
        signals.append(
            {
                "timestamp": ts,
                "security_id": security["security_id"],
                "signal_type": "long",
                "strength": 1.0,
                "confidence": 0.8,
                "meta": {"note": "example signal"},
            }
        )
    output = {"model_id": payload["model_id"], "run_id": payload["run_id"], "signals": signals}
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()

# modules/usage_tracker.py
# Version 1.0.0: Initial implementation of daily usage tracking for Google API.

import json
import os
from datetime import datetime
from typing import Dict, Any

USAGE_FILE = "usage_stats.json"
DAILY_LIMIT = 100

def _load_stats() -> Dict[str, Any]:
    """Loads stats from the JSON file or returns a default if it doesn't exist."""
    default_stats = {"count": 0, "date": str(datetime.now().date())}
    
    if not os.path.exists(USAGE_FILE):
        return default_stats
    
    try:
        with open(USAGE_FILE, 'r') as f:
            stats = json.load(f)
            # Check if date is today, if not reset
            today = str(datetime.now().date())
            if stats.get("date") != today:
                return default_stats
            return stats
    except (json.JSONDecodeError, IOError):
        return default_stats

def _save_stats(stats: Dict[str, Any]):
    """Saves stats to the JSON file."""
    try:
        with open(USAGE_FILE, 'w') as f:
            json.dump(stats, f)
    except IOError:
        pass

def get_usage() -> int:
    """Returns the current usage count for today."""
    stats = _load_stats()
    return stats.get("count", 0)

def increment_usage(amount: int = 1):
    """Increments the daily usage count."""
    stats = _load_stats()
    stats["count"] = stats.get("count", 0) + amount
    _save_stats(stats)

def get_limit() -> int:
    """Returns the daily limit."""
    return DAILY_LIMIT

# // end of modules/usage_tracker.py

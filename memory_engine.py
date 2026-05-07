import math
from datetime import datetime, timedelta


# =========================
# MEMORY RETENTION FUNCTION
# =========================
def retention(t, S):
    """
    Returns memory retention after time t (in hours)
    S = memory strength
    """
    return math.exp(-t / S)


# =========================
#  NEXT REVISION CALCULATION
# =========================
def next_revision(current_time, strength, memory_type="average"):
    """
    Calculates next revision time based on forgetting curve.

    Parameters:
    - current_time: datetime.now()
    - strength: base memory strength (e.g., 2.0)
    - memory_type: forget_fast / average / strong
    """

    #  Target retention threshold
    target_retention = 0.6

    # =========================
    #  MAP UI → MEMORY BEHAVIOR
    # =========================
    if memory_type == "forget_fast":
        S = strength * 1.0   # fast decay
    elif memory_type == "strong":
        S = strength * 3.0   # slow decay
    else:
        S = strength * 2.0   # normal

    # =========================
    #  SOLVE EXPONENTIAL DECAY
    # exp(-t/S) = target_retention
    # =========================
    t = -S * math.log(target_retention)

    # convert to datetime
    next_time = current_time + timedelta(hours=t)

    return next_time
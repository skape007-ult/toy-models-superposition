import numpy as np
def mode_axis0(a):
    """Majority vote along axis 0 for small integer arrays (ties -> smallest label)."""
    a = np.asarray(a)
    k = a.max() + 1
    counts = np.stack([(a == c).sum(0) for c in range(k)], 0)
    return counts.argmax(0)

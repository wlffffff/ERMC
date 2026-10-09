"""Type-compatible goals; no coordinates, hidden map, or object identities."""
import numpy as np
ACTIVE_STAGES = 3
STATE_COUNT = ACTIVE_STAGES + 1
OBJECTS = ('direct', 'key', 'door', 'ball', 'box')
PREDICATES = ('direct', 'visible', 'ready', 'acquired', 'opened')
OBJECT_CODES = (0, 5, 4, 6, 7)
GOALS = [(0, 0)] + [(o, p) for o in range(1, 5) for p in (1, 2)]
GOALS += [(o, 3) for o in (1, 3, 4)] + [(2, 4)]
GOALS = tuple(GOALS)
N_GOALS = len(GOALS)
FIXED_IDS = (GOALS.index((1, 1)), GOALS.index((1, 2)))
GOAL_FEATURES = np.zeros((N_GOALS, 10), dtype=np.float32)
for i, (o, p) in enumerate(GOALS[1:], 1):
    GOAL_FEATURES[i, o] = GOAL_FEATURES[i, 5 + p] = 1.0
CONTEXT_DIM = 12 + N_GOALS + 4 + STATE_COUNT

def name(goal):
    o, p = GOALS[goal]
    return OBJECTS[o] + '_' + PREDICATES[p]

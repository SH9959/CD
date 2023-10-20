from .linear import Notears
from .low_rank import NotearsLowRank

from trustworthyAI.gcastle.castle.backend import backend

if backend == 'pytorch':
    from .torch import NotearsNonlinear
    from .torch import GOLEM

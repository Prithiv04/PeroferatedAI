import copy
from . import globals_perforatedai as GPA

def deep_copy_pai(module):
    """
    Custom deep copy for PAI modules that clears processors before copying.
    """
    if hasattr(GPA, 'pai_tracker') and hasattr(GPA.pai_tracker, 'clear_all_processors'):
        GPA.pai_tracker.clear_all_processors()
    return copy.deepcopy(module)

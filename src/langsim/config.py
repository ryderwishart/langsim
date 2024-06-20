"""
Global debug mode for the langsim package.

FIXME: not working as expected if user uses `set_debug_mode` to set debug mode.
Currently, user must pass `debug=True` to `compare_languages` or `view_pairwise_scores`
to enable debug mode.
"""

DEBUG_MODE = None

def set_debug_mode(debug: bool):
    global DEBUG_MODE
    DEBUG_MODE = debug

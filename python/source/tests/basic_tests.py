
import numpy as np
import traceback


def run_basic_test(istar_class, logger):

    all_okay = True
    log = "---BASIC TEST RESULTS---\n\n"

    log += "     1. Data dimensions: 1, Base functions: 1\n"
    data_dimensions = 1
    base_functions = 1
    try:
        my_istar = istar_class(logger = logger, data_dimensions = data_dimensions, base_functions = base_functions)
        my_istar.init_weights("common",10)
        my_istar.init_weights("random",10)
        my_istar.init_weights("external",np.ones((base_functions,data_dimensions)))
    except Exception as e:
        all_okay = False
        log += "     error: " + str(e)
        log += "\n     traceback: " + traceback.format_exc()
        return all_okay, log

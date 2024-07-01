
import numpy as np
import traceback
from ..core.data_analysis import *


def run_basic_test(istar_class, logger):

    all_okay = True
    log = "---BASIC TEST RESULTS---\n\n"

    title = "     1. Data dimensions: 1, Base functions: 1 |     "
    adj = 100 - len(title)
    log += title
    data_dimensions = 1
    base_functions = 1
    try:
        my_istar = istar_class(logger = logger, data_dimensions = data_dimensions, base_functions = base_functions)
        my_istar.init_weights(method = "common", val = 10)
        my_istar.init_weights(method = "random", val = 10)
        my_istar.init_weights(method = "external", val = np.ones((base_functions,data_dimensions)))

        my_istar.init_bias(method = "lineal")
        my_istar.init_bias(method = "random", val = 0.5)
        my_istar.init_bias(method = "external", val = np.ones((base_functions,data_dimensions)))

        log += adj*"-" +"PASS\n"
    except Exception as e:
        all_okay = False
        log += adj*"-" +"FAILED\n\n"
        log += "     error: " + str(e)
        log += "\n\n     traceback: " + traceback.format_exc()
        return all_okay, log

    title = "     2. Data dimensions: 10, Base functions: 20 |     "
    adj = 100 - len(title)
    log += title
    data_dimensions = 10
    base_functions = 20
    try:
        my_istar = istar_class(logger = logger, data_dimensions = data_dimensions, base_functions = base_functions)
        my_istar.init_weights(method = "common", val = 10)
        my_istar.init_weights(method = "random", val = 10)
        my_istar.init_weights(method = "external", val = np.ones((base_functions,data_dimensions)))

        my_istar.init_bias(method = "lineal")
        my_istar.init_bias(method = "random", val = 0.5)
        my_istar.init_bias(method = "external", val = np.ones((base_functions,data_dimensions)))

        log += adj*"-" +"PASS\n"
    except Exception as e:
        all_okay = False
        log += adj*"-" +"FAILED\n\n"
        log += "     error: " + str(e)
        log += "\n\n     traceback: " + traceback.format_exc()

    log += "\n\n"
    return all_okay, log

def run_core_test(istar_class, logger):

    all_okay = True
    log = "---CORE TEST RESULTS---\n\n"

    title = "     1. Bias lineal initialization, Base functions: 100 |     "
    adj = 100 - len(title)
    log += title
    data_dimensions = 1
    base_functions = 100

    ## Define DATA
    n_data = 500
    r_data = 1/n_data
    data_in = np.ones((n_data,data_dimensions))
    data_area = r_data*np.ones((n_data,data_dimensions))
    data_lab = np.zeros(n_data)
    for i in range(n_data):
        data_in[i,:] = (2*i/n_data-1)*np.ones(data_dimensions)
        data_lab[i] = np.sin(10*np.sum(data_in[i,:]))

    try:
        my_istar = istar_class(logger = logger, data_dimensions = data_dimensions, base_functions = base_functions)
        my_istar.init_weights(method = "common", val = 100)
        my_istar.init_bias(method = "lineal")
        my_istar.evaluate_ortogonal_base()
        my_istar.evaluate_proyection(data_in, data_area, data_lab)
        output = my_istar.run(data_in)
        error = np.mean(np.abs(data_lab - output))/2*100

        if(np.isnan(error).any()):
            raise ValueError("NAN error")

        log += adj*"-" +"PASS\n"
        log += "        score (less is better) = " + str(error) + "/1.613\n"
    except Exception as e:
        all_okay = False
        log += adj*"-" +"FAILED\n"
        log += "     error: " + str(e)
        log += "\n     traceback: " + traceback.format_exc()
    log += "\n\n"
    return all_okay, log

def run_distribution_test(istar_class, logger, base_functions = 100, w_coef = 0.5):

    all_okay = True
    log = "---DISTRIBUTION TEST RESULTS---\n\n"
    data_dimensions = 1

    ## Define DATA
    n_data = 500
    r_data = 1/n_data
    data_in = np.ones((n_data,data_dimensions))
    data_area = r_data*np.ones((n_data,data_dimensions))
    data_lab = np.zeros(n_data)
    for i in range(n_data):
        data_in[i,:] = (2*i/n_data-1)*np.ones(data_dimensions)
        data_lab[i] = np.sin(10*np.sum(data_in[i,:]))

    title = "     1. Bias data distribution initialization, Base functions: " + str(base_functions) + " |     "
    adj = 100 - len(title)
    log += title
    try:
        my_istar = istar_class(logger = logger, data_dimensions = data_dimensions, base_functions = base_functions)

        (bias,weights) = distribute_bases(data_in, base_functions, w_coef)
        my_istar.init_weights(method = "common", val = 100)
        my_istar.init_bias(method = "external", val = bias)

        my_istar.evaluate_ortogonal_base()
        my_istar.evaluate_proyection(data_in, data_area, data_lab)
        output = my_istar.run(data_in)
        error = np.mean(np.abs(data_lab - output))/2*100

        if(np.isnan(error).any()):
            raise ValueError("NAN error")

        log += adj*"-" +"PASS\n"
        log += "        score (less is better) = " + str(error) + "/1.613\n"
    except Exception as e:
        all_okay = False
        log += adj*"-" +"FAILED\n"
        log += "     error: " + str(e)
        log += "\n     traceback: " + traceback.format_exc()
        log += "\n"

    title = "     2. Weights data distribution initialization, Base functions: " + str(base_functions) + " |     "
    adj = 100 - len(title)
    log += title
    try:
        my_istar = istar_class(logger = logger, data_dimensions = data_dimensions, base_functions = base_functions)

        (bias,weights) = distribute_bases(data_in, base_functions, w_coef)
        my_istar.init_weights(method = "external", val = weights)
        my_istar.init_bias(method = "lineal")

        my_istar.evaluate_ortogonal_base()
        my_istar.evaluate_proyection(data_in, data_area, data_lab)
        output = my_istar.run(data_in)
        error = np.mean(np.abs(data_lab - output))/2*100

        if(np.isnan(error).any()):
            raise ValueError("NAN error")

        log += adj*"-" +"PASS\n"
        log += "        score (less is better) = " + str(error) + "/1.613\n"
    except Exception as e:
        all_okay = False
        log += adj*"-" +"FAILED\n"
        log += "     error: " + str(e)
        log += "\n     traceback: " + traceback.format_exc()
        log += "\n"

    title = "     3. Bias & Weights data distribution initialization, Base functions: " + str(base_functions) + " |     "
    adj = 100 - len(title)
    log += title
    try:
        my_istar = istar_class(logger = logger, data_dimensions = data_dimensions, base_functions = base_functions)

        (bias,weights) = distribute_bases(data_in, base_functions, w_coef)
        my_istar.init_weights(method = "external", val = weights)
        my_istar.init_bias(method = "external", val = bias)

        my_istar.evaluate_ortogonal_base()
        my_istar.evaluate_proyection(data_in, data_area, data_lab)
        output = my_istar.run(data_in)
        error = np.mean(np.abs(data_lab - output))/2*100

        if(np.isnan(error).any()):
            raise ValueError("NAN error")

        log += adj*"-" +"PASS\n"
        log += "        score (less is better) = " + str(error) + "/1.613\n"
    except Exception as e:
        all_okay = False
        log += adj*"-" +"FAILED\n"
        log += "     error: " + str(e)
        log += "\n     traceback: " + traceback.format_exc()
        log += "\n"

    title = "     4. Bias & Weights & Areas data distribution initialization, Base functions: " + str(base_functions) + " |     "
    adj = 100 - len(title)
    log += title
    try:
        my_istar = istar_class(logger = logger, data_dimensions = data_dimensions, base_functions = base_functions)

        (bias,weights,anlz_area) = analyze_data(data_in, base_functions, w_coef)
        my_istar.init_weights(method = "external", val = weights)
        my_istar.init_bias(method = "external", val = bias)

        my_istar.evaluate_ortogonal_base()
        my_istar.evaluate_proyection(data_in, anlz_area, data_lab)
        output = my_istar.run(data_in)
        error = np.mean(np.abs(data_lab - output))/2*100

        if(np.isnan(error).any()):
            raise ValueError("NAN error")

        log += adj*"-" +"PASS\n"
        log += "        score (less is better) = " + str(error) + "/1.613\n"
    except Exception as e:
        all_okay = False
        log += adj*"-" +"FAILED\n"
        log += "     error: " + str(e)
        log += "\n     traceback: " + traceback.format_exc()


    log += "\n\n"
    return all_okay, log

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from source.common.logger import create_logger
from source.core.schmidt import Istar
from source.tests.basic_tests import *
from python.source.common.plots import dibujar_vectores
from python.source.common.evaluations import evaluate_base,evaluate_orts




log = create_logger()

(all_okay, log_msg) = run_basic_test(Istar, log)
if all_okay == True:
    log.info(log_msg)
else:
    log.warning(log_msg)
    exit()

(all_okay, log_msg) = run_core_test(Istar, log)
if all_okay == True:
    log.info(log_msg)
else:
    log.warning(log_msg)
    exit()

w_coef = 0.8
for n in range(10,80,10):
    (all_okay, log_msg) = run_distribution_test(Istar, log, n, w_coef)
    if all_okay == True:
        log.info(log_msg)
    else:
        log.warning(log_msg)
        exit()

n = 50
for w_coef in np.linspace(0.9,0.1,9):
    (all_okay, log_msg) = run_distribution_test(Istar, log, n, w_coef)
    if all_okay == True:
        log.info(log_msg)
    else:
        log.warning(log_msg)
        exit()

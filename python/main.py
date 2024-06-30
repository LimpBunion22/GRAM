import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from source.common.logger import create_logger
from source.core.schmidt import Istar
from source.tests.basic_tests import *
from source.plots import dibujar_vectores
from source.evaluations import evaluate_base,evaluate_orts




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

for n in range(5,80,5):
    (all_okay, log_msg) = run_distribution_test(Istar, log, n)
    if all_okay == True:
        log.info(log_msg)
    else:
        log.warning(log_msg)
        exit()

import argparse
import os
import logging
from datetime import datetime

from solvers.solver_cls import SolverCls


ROOT_DIR = os.path.join(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(ROOT_DIR)


def cls_train():
    solver = SolverCls()
    solver.fit()
    solver.save_net('output')


def cls_test():
    solver = SolverCls()
    solver.load_net('output')
    solver.eval()


if __name__ == '__main__':
    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    # positional arguments
    parser.add_argument('mode', type=str, choices=['cls_train', 'cls_test'])

    args = parser.parse_args()
    CFG = vars(args)

    # ------
    # Setting Root Logger
    # -----
    level = logging.INFO
    format = '%(asctime)s: %(message)s'
    log_dir = os.path.join(ROOT_DIR, 'output', 'log')
    log_path = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S.log'))
    os.makedirs(log_dir, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode='w')
    ]
    logging.basicConfig(format=format, level=level, handlers=handlers)

    # -------
    # Run
    # -------
    if CFG['mode'] == 'cls_train':
        cls_train()
    elif CFG['mode'] == 'cls_test':
        cls_test()

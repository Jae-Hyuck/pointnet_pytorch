import argparse
import os
import logging
from datetime import datetime

from solvers.solver_cls import SolverCls
from solvers.solver_seg import SolverSeg
# from solvers.solver_part import SolverPart


ROOT_DIR = os.path.join(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(ROOT_DIR)


def cls_train():
    solver = SolverCls()
    solver.fit()
    solver.save_net('output')


def cls_test():
    solver = SolverCls()
    solver.load_net('output')
    solver.evaluate()


def seg_train():
    solver = SolverSeg()
    solver.fit()
    solver.save_net('output')


def seg_test():
    solver = SolverSeg()
    solver.load_net('output')
    solver.evaluate()


def part_train():
    solver = SolverPart()
    solver.fit()
    solver.save_net('output')


def part_test():
    solver = SolverPart()
    solver.load_net('output')
    solver.evaluate()


if __name__ == '__main__':
    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    # positional arguments
    parser.add_argument('mode', type=str, choices=['cls_train', 'cls_test', 'seg_train',
                                                   'seg_test', 'part_train', 'part_test'])

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
    func = globals()[CFG['mode']]
    func()

import numpy as np
import os
import sys
import logging

def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x / warmup
    return max((x - 1.0) / (warmup - 1.0), 0)


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

def get_logger(log_dir, log_name, level=logging.INFO):
    file = os.path.join(log_dir, log_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if os.path.exists(file):
        os.remove(file)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=level)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=level)
    logger.addHandler(stream_handler)
    # FileHandler
    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    meter = AverageMeter()

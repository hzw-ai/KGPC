import torch
import random
import numpy as np
import os
import errno
import yaml
import argparse
import datetime


def set_random_seed(seed=0):
    """Set random/np.random/torch.random/cuda.random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    print(f"Random seed set as {seed}")


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def setup(args):
    set_random_seed(args['seed'])
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


class EarlyStopping(object):
    def __init__(self, dataset, patience=10, log_dir=None, filename=None):
        """
        早停止控制
        :param dataset:     数据集名称
        :param patience:    耐心度，最多几轮没有变得更好就停止
        :param log_dir:     模型保存地址
        :param filename:    模型文件名
        """
        if not filename:
            filename = 'best_model.pth'
        dt = datetime.datetime.now()
        if log_dir:
            self.filename = f'{log_dir}/{filename}'
        else:
            if not os.path.exists('results/'):
                mkdir_p('results')
            self.filename = 'results/early_stop_{}_{}_{:02d}-{:02d}-{:02d}.pth'.format(dataset, dt.date(), dt.hour,
                                                                                       dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_epoch = None
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, epoch, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.best_epoch = epoch
            self.save_checkpoint(model)

        if (loss <= self.best_loss) and (acc >= self.best_acc):
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
            self.best_epoch = epoch
            self.save_checkpoint(model)
        else:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        # if (loss > self.best_loss) and (acc < self.best_acc):
        #     self.counter += 1
        #     # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        #
        # else:
        #     if (loss <= self.best_loss) and (acc >= self.best_acc):
        #         self.best_epoch = epoch
        #         self.save_checkpoint(model)
        #     self.best_loss = np.min((loss, self.best_loss))
        #     self.best_acc = np.max((acc, self.best_acc))
        #     self.counter = 0

        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))

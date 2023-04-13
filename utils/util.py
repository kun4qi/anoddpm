import os
import json
import random
import collections
import time
from collections import defaultdict
import fsspec
from pathlib import Path
from typing import Optional
from typing import Union
from typing import Dict
import datetime
import numpy as np
import torch
from torchvision.utils import save_image
import torch.nn as nn
from torch.autograd import Variable

import pytorch_lightning
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import ModelCheckpoint


def as_numpy(tensor, first_elem):
    if first_elem:
        return tensor.detach().cpu().numpy()[0, 0, ...]
    else:
        return tensor.detach().cpu().numpy()


def load_json(path):
    def _json_object_hook(d):
        return collections.namedtuple('X', d.keys())(*d.values())
    def _json_to_obj(data):
        return json.loads(data, object_hook=_json_object_hook)
    return _json_to_obj(open(path).read())


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    return seed


def load_model(model, save_path):
    model_state = torch.load(save_path)
    if isinstance(model_state, nn.DataParallel):
        model_state = model_state.module.state_dict()
    else:
        model_state = model_state.state_dict()
    model.load_state_dict(model_state)


def norm(x):
    x = 2.0 * (x - 0.5)
    return x.clamp_(-1, 1)


def denorm(x):
    x = (x + 1) / 2.0
    return x.clamp_(0, 1)


def minmax_norm(x):
    vmax = np.max(x)
    vmin = np.min(x)
    x -= vmin
    x /= (vmax - vmin)
    return x


def minmax_norm(x):
    vmax = np.max(x)
    vmin = np.min(x)
    x -= vmin
    x /= (vmax - vmin)
    return x


def calc_latent_dim(config):
    return (
        config.dataset.batch_size,
        config.model.z_dim,
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters))),
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters)))
    )


class OneHotEncoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.ones = torch.sparse.torch.eye(n_classes).cuda()

    def forward(self, t):
        n_dim = t.dim()
        output_size = t.size() + torch.Size([self.n_classes])
        t = t.data.long().contiguous().view(-1).cuda()
        out = Variable(self.ones.index_select(0, t)).view(output_size)
        out = out.permute(0, -1, *range(1, n_dim)).float()
        return out


pathlike = Union[Path, str]

def get_filesystem(path: pathlike):
    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0])
    else:
        # use local filesystem
        return fsspec.filesystem("file")


class Logger(LightningLoggerBase):

    def __init__(self,
                 save_dir: str,
                 config: collections.defaultdict,
                 seed: int,
                 monitoring_metrics: list,
                 name: Optional[str]='default',
                 version: Optional[Union[int, str]] = None,
                 **kwargs) -> None:
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self._config = config
        self._seed = seed
        self._version = version
        self._fs = get_filesystem(save_dir)
        self._experiment = None
        self._monitoring_metrics = monitoring_metrics
        self._kwargs = kwargs

    @property
    def root_dir(self) -> str:
        if self.name is None or len(self.name) == 0:
            return self.save_dir
        else:
            return os.path.join(self.save_dir, self.name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def log_dir(self) -> str:
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def version(self) -> int:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = os.path.join(self.save_dir, self.name)

        if not self._fs.isdir(root_dir):
            print('Missing logger folder: %s', root_dir)
            return 0

        existing_versions = []
        for listing in self._fs.listdir(root_dir):
            d = listing["name"]
            bn = os.path.basename(d)
            if self._fs.isdir(d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace('/', '')
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        values = []
        for key in self._monitoring_metrics:
            if key in metrics.keys():
                v = metrics[key]
                if isinstance(v, torch.Tensor):
                    v = str(v.item())
                else:
                    v = str(v)
            else:
                v = ''
            values.append(v)

        fname = os.path.join(self.log_dir, 'log.csv')
        with open(fname, 'a') as f:
            if f.tell() == 0:
                print(','.join(self._monitoring_metrics), file=f)
            print(','.join(values), file=f)

    @rank_zero_only
    def log_val_metrics(self, metrics: Dict[str, float]) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        fname = os.path.join(self.log_dir, 'val_logs.csv')
        columns = metrics.keys()
        values = []
        for key in self._monitoring_metrics:
            if key in metrics.keys():
                v = metrics[key]
                v = str(v)
            else:
                v = ''
            values.append(v)

        with open(fname, 'a') as f:
            if f.tell() == 0:
                print(','.join(self._monitoring_metrics), file=f)
            print(','.join(values), file=f)

    @rank_zero_only
    def log_test_metrics(self, metrics: Dict[str, float]) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        fname = os.path.join(self.log_dir, 'test_logs.csv')

        for i in range(len(metrics)):
            columns = metrics[i].keys()
            values = [str(value) for value in metrics[i].values()]

            with open(fname, 'a') as f:
                if f.tell() == 0:
                    print(','.join(columns), file=f)
                print(','.join(values), file=f)
        
        print('Test results are saved: {}'.format(fname))

    @rank_zero_only
    def log_hyperparams(self, config, needs_save):
        config_to_save = defaultdict(dict)
        for key, child in config._asdict().items():
            for k, v in child._asdict().items():
                config_to_save[key][k] = v

        config_to_save['seed'] = config.training.seed
        config_to_save['output_dir'] = self.log_dir

        print('Training starts by the following configuration: ', config_to_save)

        if needs_save:
            save_path = os.path.join(self.log_dir, 'config.json')
            with open(save_path, 'w') as f:
                json.dump(config_to_save, f)

    @rank_zero_only
    def val_log_images(self, image: torch.Tensor, current_epoch: int, nrow: int) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'
        save_path = os.path.join(self.log_dir, 'val_result_{}.png'.format(current_epoch))
        save_image(image.data, save_path, nrow=nrow)

    @rank_zero_only
    def train_log_images(self, image: torch.Tensor, current_epoch: int, nrow: int) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'
        save_path = os.path.join(self.log_dir, 'train_result_{}.png'.format(current_epoch))
        save_image(image.data, save_path, nrow=nrow)

    def experiment(self):
        return self

    def save(self):
        super().save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()

class Time(Callback):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.config.training.use_cuda:
            torch.cuda.synchronize()
            self.start = time.time()

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        elapsed_time = time.time() - self.start
        print()
        print(f'Time per epoch: {elapsed_time:0.3f}[s]')
        
        

        
class ModelSaver(ModelCheckpoint):

    def __init__(self, limit_num, save_interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit_num = limit_num
        self.save_interval = save_interval

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the training epoch."""
        if not self._should_skip_saving_checkpoint(trainer) and self._save_on_train_epoch_end:
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

            if trainer.is_global_zero:
                self._delete_old_checkpoint(trainer)

    def _delete_old_checkpoint(self, trainer):
        checkpoints = sorted([c for c in os.listdir(self.dirpath) if 'ckpt-epoch' in c])

        if len(checkpoints) > self.limit_num:
            margin = len(checkpoints) - self.limit_num
            checkpoints_for_delete = checkpoints[:margin]

            for ckpt in checkpoints_for_delete:
                ckpt_epoch = int(ckpt[len("ckpt-epoch="): len("ckpt-epoch=") + 4])
                if (ckpt_epoch + 1) % self.save_interval != 0:
                    model_path = os.path.join(self.dirpath, ckpt)
                    os.remove(model_path)
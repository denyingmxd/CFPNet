import matplotlib.cm
import torch
import torch.nn

from src.models.deltar import Deltar
from src.models.deltar_rgb import Deltar_rgb
from src.models.deltar_no_refine import Deltar_no_refine
from src.models.deltar_residual_1 import Deltar_residual_1
from src.models.deltar_residual_2 import Deltar_residual_2
from src.models.deltar_residual_3 import Deltar_residual_3
from src.models.deltar_residual_4 import Deltar_residual_4
from src.models.deltar_refine_1 import Deltar_refine_1
from src.models.deltar_refine_2 import Deltar_refine_2
from src.models.deltar_refine_3 import Deltar_refine_3
from src.models.deltar_refine_4 import Deltar_refine_4
def make_model(args):
    if args.model_name=='deltar':
        model = Deltar(n_bins=args.n_bins, min_val=args.min_depth,
                        max_val=args.max_depth, norm=args.norm)
    elif args.model_name=='deltar_residual_1':
        model = Deltar_residual_1(n_bins=args.n_bins, min_val=args.min_depth,
                        max_val=args.max_depth, norm=args.norm)
    elif args.model_name=='deltar_residual_2':
        model = Deltar_residual_2(n_bins=args.n_bins, min_val=args.min_depth,
                        max_val=args.max_depth, norm=args.norm)
    elif args.model_name=='deltar_residual_3':
        model = Deltar_residual_3(n_bins=args.n_bins, min_val=args.min_depth,
                        max_val=args.max_depth, norm=args.norm)
    elif args.model_name=='deltar_residual_4':
        model = Deltar_residual_4(n_bins=args.n_bins, min_val=args.min_depth,
                        max_val=args.max_depth, norm=args.norm)
    elif args.model_name=='deltar_rgb':
        model = Deltar_rgb(n_bins=args.n_bins, min_val=args.min_depth,
                       max_val=args.max_depth, norm=args.norm)
    elif args.model_name == 'deltar_no_refine':
        model = Deltar_no_refine(n_bins=args.n_bins, min_val=args.min_depth,
                           max_val=args.max_depth, norm=args.norm,d_type=args.d_type)
    elif args.model_name == 'deltar_refine_1':
        model = Deltar_refine_1(n_bins=args.n_bins, min_val=args.min_depth,
                                 max_val=args.max_depth, norm=args.norm)
    elif args.model_name == 'deltar_refine_2':
        model = Deltar_refine_2(n_bins=args.n_bins, min_val=args.min_depth,
                                 max_val=args.max_depth, norm=args.norm)
    elif args.model_name == 'deltar_refine_3':
        model = Deltar_refine_3(n_bins=args.n_bins, min_val=args.min_depth,
                                 max_val=args.max_depth, norm=args.norm)
    elif args.model_name == 'deltar_refine_4':
        model = Deltar_refine_4(n_bins=args.n_bins, min_val=args.min_depth,
                                 max_val=args.max_depth, norm=args.norm)
    return model


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


def colorize(value, vmin=10, vmax=1000, cmap='magma_r'):
    value = value.cpu().numpy()[0, :, :]
    invalid_mask = value == -1

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img



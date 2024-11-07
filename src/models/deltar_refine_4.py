import torch
import torch.nn as nn


from .CompletetionModel2 import DepthCompletion2

class Deltar_refine_4(nn.Module):
    def __init__(self, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(Deltar_refine_4, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.completionnet = DepthCompletion2()



    def forward(self, input_data, pred1, prob):

        pred2 = self.completionnet(input_data['rgb'],pred1)

        return pred1, pred2, prob


    def get_1x_lr_params(self):
        return self.completionnet.parameters()


if __name__ == '__main__':
    model = Deltar.build(100)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    print(bins.shape, pred.shape)

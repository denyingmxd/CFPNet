import torch
import torch.nn as nn
import torch.nn.functional as F
class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)



class MyLoss1(nn.Module):
    def __init__(self):
        super(MyLoss1, self).__init__()
        self.name = 'MyLoss1'

    def forward(self, input, target, my_mask, sample_num=1000, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
        B,C,H,W = input.shape
        loss = 0
        for i in range(input.shape[0]):
            local_input = input[i]
            local_target = target[i]
            inside_area = my_mask[i] * mask[i]
            outside_area = (~my_mask[i]) * mask[i]
            inside_input = torch.masked_select(local_input, inside_area)
            outside_input = torch.masked_select(local_input, outside_area)
            inside_target = torch.masked_select(local_target, inside_area)
            outside_target = torch.masked_select(local_target, outside_area)

            num_inside = inside_area.sum()
            num_outside = mask[i].sum()- num_inside
            #randomly select 1000 pixels from inside
            if num_inside > sample_num:
                inside_input = inside_input[torch.randperm(num_inside)[:sample_num]]
                inside_target = inside_target[torch.randperm(num_inside)[:sample_num]]
            #randomly select 1000 pixels from outside
            if num_outside > sample_num:
                outside_input = outside_input[torch.randperm(num_outside)[:sample_num]]
                outside_target = outside_target[torch.randperm(num_outside)[:sample_num]]
            #calculate pairwise distance for input inside and outside
            pd_input = torch.cdist(inside_input.unsqueeze(-1), outside_input.unsqueeze(-1), p=2)
            #calculate pairwise distance for target inside and outside
            pd_target = torch.cdist(inside_target.unsqueeze(-1), outside_target.unsqueeze(-1), p=2)
            #calculate loss
            loss += torch.abs(pd_input-pd_target).mean()

        return loss

class MyLoss2(nn.Module):
    def __init__(self):
        super(MyLoss2, self).__init__()
        self.name = 'MyLoss2'
    def forward(self, input, target, my_mask, sample_num=1000, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
        B,C,H,W = input.shape
        loss = 0
        for i in range(input.shape[0]):
            local_input = input[i]
            local_target = target[i]
            inside_area = my_mask[i] * mask[i]
            outside_area = (~my_mask[i]) * mask[i]
            inside_input = torch.masked_select(local_input, inside_area)
            outside_input = torch.masked_select(local_input, outside_area)
            inside_target = torch.masked_select(local_target, inside_area)
            outside_target = torch.masked_select(local_target, outside_area)

            num_inside = inside_area.sum()
            num_outside = mask[i].sum()- num_inside

            if num_inside > sample_num:
                inside_input = inside_input[torch.randperm(num_inside)[:sample_num]]
                inside_target = inside_target[torch.randperm(num_inside)[:sample_num]]
            #randomly select 1000 pixels from outside
            if num_outside > sample_num:
                outside_input = outside_input[torch.randperm(num_outside)[:sample_num]]
                outside_target = outside_target[torch.randperm(num_outside)[:sample_num]]

            #calculate inner product between inside and outside for input
            ip_input = F.softmax(torch.mm(inside_input.unsqueeze(-1), outside_input.unsqueeze(-1).t()), dim=1)
            #for target
            ip_target = F.softmax(torch.mm(inside_target.unsqueeze(-1), outside_target.unsqueeze(-1).t()),dim=1)

            loss += torch.abs(ip_input-ip_target).sum()/1000.

        return loss

class MyLoss3(nn.Module):
    def __init__(self):
        super(MyLoss3, self).__init__()
        self.name = 'MyLoss3'

    def forward(self, input, target, my_mask, sample_num=1000, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
        B,C,H,W = input.shape
        loss = 0
        for i in range(input.shape[0]):
            local_input = input[i]
            local_target = target[i]
            inside_area = my_mask[i] * mask[i]
            outside_area = (~my_mask[i]) * mask[i]
            inside_input = torch.masked_select(local_input, inside_area)
            outside_input = torch.masked_select(local_input, outside_area)
            inside_target = torch.masked_select(local_target, inside_area)
            outside_target = torch.masked_select(local_target, outside_area)

            num_inside = inside_area.sum()
            num_outside = mask[i].sum()- num_inside
            #randomly select 1000 pixels from inside
            if num_inside > sample_num:
                inside_input = inside_input[torch.randperm(num_inside)[:sample_num]]
                inside_target = inside_target[torch.randperm(num_inside)[:sample_num]]
            #randomly select 1000 pixels from outside
            if num_outside > sample_num:
                outside_input = outside_input[torch.randperm(num_outside)[:sample_num]]
                outside_target = outside_target[torch.randperm(num_outside)[:sample_num]]
            #calculate pairwise distance for input inside and outside
            pd_input = torch.cdist(inside_input.unsqueeze(-1), outside_input.unsqueeze(-1), p=2)
            #calculate pairwise distance for target inside and outside
            pd_target = torch.cdist(inside_target.unsqueeze(-1), outside_target.unsqueeze(-1), p=2)
            #calculate loss
            loss += torch.abs(pd_input-pd_target).mean()

        return loss
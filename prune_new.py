from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from torchvision.models.resnet import BasicBlock, Bottleneck
from typing import List
import warnings

torch.set_printoptions(sci_mode=False)


class ConvBn():
    def __init__(self, conv_module: nn.Conv2d, bn_module: nn.BatchNorm2d, conv_index: int, bn_index: int,
                 downsample: nn.Sequential = None) -> None:
        assert isinstance(conv_module, nn.Conv2d) and isinstance(bn_module, nn.BatchNorm2d)

        self.conv = conv_module
        self.bn = bn_module
        self.input_mask = None
        self.output_mask = None
        self.conv_index = conv_index
        self.bn_index = bn_index

        self.downsample = downsample
        self.downsample_input_mask = None
        self.downsample_output_mask = None

    def __repr__(self) -> str:
        str_format = "conv: {}\tbn: {}\tconv_index: {}\tbn_index: {}\thas downsample: {}"
        return str_format.format(self.conv, self.bn, self.conv_index, self.bn_index, self.downsample is not None)

    def add_downsample(self, downsample: nn.Sequential):
        assert self.bn.weight.size() == downsample[1].weight.size()
        self.downsample = downsample


class PruneTool:
    '''
    currently, only for normal Block, such as BottleNeck and BasicBlock......

    can prune downsample
    '''

    def __init__(self, percentage: float, model: nn.Module, devices='cpu', block: nn.Module = BasicBlock, channel_limit=8, fuse_method=0):

        self.prune_keep_size_model = deepcopy(model).eval()
        self.percentage = percentage
        self.devices = devices
        self.prune_block = block
        self.channel_limit = channel_limit
        self.fuse_method = fuse_method

        self.bn_thrsh = -1
        self.block_length = -1

        self.been_pruned = False

        self.model_list = self.get_model_list(self.prune_keep_size_model)
        self.compute_mask()

    def get_dummy_downsample(self, in_channel: int, out_channel: int):
        '''
        如果是使用了设个函数创建downsample，表示原BottleNeck是恒等变换，因此必然成立
        '''
        assert in_channel == out_channel
        downsample = nn.Sequential(nn.Conv2d(in_channel, out_channel, (1, 1), (1, 1), (0, 0), bias=False),
                                   nn.BatchNorm2d(out_channel)).to(self.devices)
        nn.init.dirac_(downsample[0].weight.data)
        nn.init.zeros_(downsample[1].running_mean.data)
        nn.init.zeros_(downsample[1].bias.data)
        nn.init.ones_(downsample[1].running_var.data)
        nn.init.ones_(downsample[1].weight.data)
        downsample.eval()
        return downsample

    def fuse_mask(self, mask1: torch.BoolTensor, mask2: torch.BoolTensor, method: int = 0):
        assert mask1.size() == mask2.size()

        if method == 0:
            mask = mask1
        elif method == 1:
            mask = mask2
        elif method == 2:
            mask = mask2 & mask1
        elif method == 3:
            mask = mask2 | mask1
        else:
            raise NotImplementedError
        return mask

    def get_model_list(self, model: nn.Module) -> List[ConvBn]:
        model_list = [module for module in model.modules() if isinstance(module, self.prune_block)]
        res_list = []

        for i, module in enumerate(model_list):
            # [conv, bn, ..., ..downsample(maybe)]
            cur_model_list = [(name, child) for (name, child) in module.named_children() if not isinstance(child, nn.ReLU)]
            assert hasattr(module, 'downsample')
            d = getattr(module, 'downsample')
            if d is None:
                in_c = cur_model_list[0][1].weight.size()[1]
                setattr(module, 'downsample', self.get_dummy_downsample(in_c, in_c))
            cur_convbn_list = []
            cur_model_list = [(name, child) for (name, child) in module.named_children() if not isinstance(child, nn.ReLU)]

            if self.block_length == -1:
                self.block_length = len(cur_model_list) // 2
            else:
                assert self.block_length == len(cur_model_list) // 2

            for i in range(0, len(cur_model_list) - 1, 2):
                struct = ConvBn(conv_module=cur_model_list[i][1], bn_module=cur_model_list[i + 1][1], conv_index=i, bn_index=i + 1)
                cur_convbn_list.append(struct)
            cur_convbn_list[-1].add_downsample(cur_model_list[-1][1])
            res_list.extend(cur_convbn_list)

        print('there are {} ConvBn block which will be pruned......'.format(len(res_list)))
        print('cur block is {}\tlength is: {}'.format(self.prune_block, self.block_length))
        return res_list

    def get_bn_thresh(self) -> None:
        weight_list = list()
        for module in self.model_list:  # [ConvBn]
            weight_list.append(module.bn.weight.data.clone().abs())
            # 所有有downsample部分，都将BN的weight取出（不管是原本的downsample，还是dummy downsample）
            if module.downsample is not None:
                weight_list.append(module.downsample[1].weight.data.clone().abs())
        weight_tensor = torch.cat(weight_list, dim=0)
        weight_tensor = weight_tensor.sort(dim=0)[0]

        index = int(weight_tensor.size(0) * self.percentage)
        thresh = float(weight_tensor[index].item())
        self.bn_thrsh = thresh
        print('bn thresh is {}......'.format(self.bn_thrsh))

    def compute_mask(self, verbose=True) -> None:
        if self.bn_thrsh < 0:
            self.get_bn_thresh()
        assert self.bn_thrsh >= 0

        input_mask = None
        output_mask = None

        for i, convbn in enumerate(self.model_list):
            o_channel = convbn.bn.weight.size()[0]
            if i == 0:
                input_mask = torch.ones(convbn.conv.weight.size()[1]).to(self.devices)
            else:
                input_mask = output_mask

            if i == len(self.model_list) - 1:
                output_mask = torch.ones(convbn.conv.weight.size()[0]).to(self.devices)
            else:
                conv_bn_output_mask = (convbn.bn.weight.data.abs().ge(self.bn_thrsh))
                if convbn.downsample is not None:
                    downsample_output_mask = (convbn.downsample[1].weight.data.abs().ge(self.bn_thrsh))
                    conv_bn_output_mask = self.fuse_mask(conv_bn_output_mask, downsample_output_mask, method=self.fuse_method)

                remain_channel = conv_bn_output_mask.sum().item()
                real_remain_channel = remain_channel // self.channel_limit * self.channel_limit + (
                    0 if remain_channel % self.channel_limit == 0 else self.channel_limit)
                real_remain_channel = max(self.channel_limit, real_remain_channel)
                if real_remain_channel == convbn.bn.weight.size()[0]:
                    output_mask = torch.ones(real_remain_channel).to(self.devices)
                else:
                    real_prune_channel = convbn.bn.weight.size()[0] - real_remain_channel
                    clone_weight = convbn.bn.weight.data.clone().abs().sort()[0]
                    real_thresh = clone_weight[real_prune_channel - 1].item()
                    output_mask = (convbn.bn.weight.data.abs().gt(real_thresh)).float()
                if verbose:
                    print('raw channel: {}\tprune channel: {}\tprune rate(cur module): {:.2f}%'.format(o_channel,
                                                                                                       o_channel - real_remain_channel, (
                                                                                                                   1. - real_remain_channel / o_channel) * 100))

            convbn.input_mask = input_mask
            convbn.output_mask = output_mask
            if convbn.downsample is not None:
                convbn.downsample_input_mask = self.model_list[i - self.block_length + 1].input_mask
                convbn.downsample_output_mask = output_mask

    def mask_model_for_prune(self):
        if self.been_pruned:
            warnings.warn('model has been pruned, not do this again......')
            return

        for i in range(len(self.model_list) - 1):
            cur_conv_bn = self.model_list[i]
            cur_output_mask = cur_conv_bn.output_mask
            cur_conv_bn.bn.weight.data.mul_(cur_output_mask)
            pre_activation = (1 - cur_output_mask) * cur_conv_bn.bn.bias.data
            if cur_conv_bn.downsample is not None:
                d = cur_conv_bn.downsample
                d[1].weight.data.mul_(cur_output_mask)
                pre_activation += (1 - cur_output_mask) * d[1].bias.data
            activation = F.relu(pre_activation)

            next_conv_bn = self.model_list[i + 1]
            conv_sum = next_conv_bn.conv.weight.data.sum(dim=(2, 3))
            offset = conv_sum.matmul(activation.view(-1, 1)).reshape(-1)
            next_conv_bn.bn.running_mean.data.sub_(offset)

            if i + self.block_length < len(self.model_list) and self.model_list[i + self.block_length].downsample is not None:
                down_conv_bn = self.model_list[i + self.block_length]
                d_conv_sum = down_conv_bn.downsample[0].weight.data.sum(dim=(2, 3))
                d_offset = d_conv_sum.matmul(activation.view(-1, 1)).reshape(-1)
                down_conv_bn.downsample[1].running_mean.data.sub_(d_offset)

            cur_conv_bn.bn.bias.data.mul_(cur_output_mask)
            if cur_conv_bn.downsample is not None:
                cur_conv_bn.downsample[1].bias.data.mul_(cur_output_mask)

    def get_compact_model(self, verbose=False) -> nn.Module:
        compact_model = deepcopy(self.prune_keep_size_model).eval()
        compact_model_list = self.get_model_list(compact_model)
        assert compact_model_list.__len__() == self.model_list.__len__()
        length = len(compact_model_list)

        # 这里有问题
        for i in range(length):
            compact_conv_bn = compact_model_list[i]
            loose_conv_bn = self.model_list[i]
            raw_size = loose_conv_bn.conv.weight.size()

            input_mask = loose_conv_bn.input_mask
            output_mask = loose_conv_bn.output_mask

            # for bn
            compact_conv_bn.bn.weight.data = loose_conv_bn.bn.weight.data[output_mask.bool()].clone()
            compact_conv_bn.bn.bias.data = loose_conv_bn.bn.bias.data[output_mask.bool()].clone()
            compact_conv_bn.bn.running_mean.data = loose_conv_bn.bn.running_mean.data[output_mask.bool()].clone()
            compact_conv_bn.bn.running_var.data = loose_conv_bn.bn.running_var.data[output_mask.bool()].clone()
            # for conv
            compact_conv_bn.conv.weight.data = loose_conv_bn.conv.weight.data[:, input_mask.bool(), ...][output_mask.bool(), ...].clone()
            new_size = compact_conv_bn.conv.weight.size()

            # for downsample
            if compact_conv_bn.downsample is not None:
                d_input_mask = loose_conv_bn.downsample_input_mask
                compact_conv_bn.downsample[0].weight.data = loose_conv_bn.downsample[0].weight.data[:, d_input_mask.bool(), ...][
                    output_mask.bool(), ...].clone()
                # for bn
                compact_conv_bn.downsample[1].weight.data = loose_conv_bn.downsample[1].weight.data[output_mask.bool()].clone()
                compact_conv_bn.downsample[1].bias.data = loose_conv_bn.downsample[1].bias.data[output_mask.bool()].clone()
                compact_conv_bn.downsample[1].running_mean.data = loose_conv_bn.downsample[1].running_mean.data[output_mask.bool()].clone()
                compact_conv_bn.downsample[1].running_var.data = loose_conv_bn.downsample[1].running_var.data[output_mask.bool()].clone()

            if verbose:
                print('weight transfer {} to {} in block[{}]'.format(raw_size, new_size, i))

        return compact_model

    def get_prune_model(self):
        return self.prune_keep_size_model

    def reset_model(self, model: nn.Module):
        self.prune_keep_size_model = deepcopy(model).eval()
        self.been_pruned = False
        self.bn_thrsh = -1
        self.block_length = -1
        print('will prune new model......')

        self.model_list = self.get_model_list(self.prune_keep_size_model)
        self.compute_mask()


def init_bn_weight(module: torch.nn.Module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data = torch.rand(m.weight.size())


def func1():
    x = torch.rand(1, 3, 224, 224)

    model = resnet50(False).eval()
    fc_size = model.fc.weight.size()
    fc = nn.Linear(fc_size[1], 5, bias=False)
    nn.init.kaiming_normal_(fc.weight)
    model.fc = fc
    print(model(x).size())

    init_bn_weight(model)
    pruneTool = PruneTool(0.1, model, block=Bottleneck)
    pruneTool.mask_model_for_prune()
    mask_model = pruneTool.prune_keep_size_model.eval()
    compact_model = pruneTool.get_compact_model(False).eval()

    model2 = resnet101(False).eval()
    fc_size = model.fc.weight.size()
    fc = nn.Linear(fc_size[1], 5, bias=False)
    nn.init.kaiming_normal_(fc.weight)
    model.fc = fc
    print(model(x).size())
    pruneTool.reset_model(model2)

    with torch.no_grad():
        y = model(x)
        y1 = mask_model(x)
        y2 = compact_model(x)
        print(y)
        print(y1)
        print(y2)


if __name__ == '__main__':
    # trainCIFAR10(device='cuda')
    func1()

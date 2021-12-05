from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from collections import OrderedDict


class ConvBn:
    def __init__(self, conv_module, bn_module, conv_index, bn_index, block_index) -> None:
        self.conv = conv_module
        self.bn = bn_module
        self.input_mask = None
        self.output_mask = None
        self.conv_index = conv_index
        self.bn_index = bn_index
        self.block_index = block_index

    def __repr__(self) -> str:
        str_format = "block index: {}\tconv: {}\tbn: {}\tconv_index: {}\tbn_index: {}"
        return str_format.format(self.block_index, self.conv, self.bn, self.conv_index, self.bn_index)

    def mask_bn(self):
        self.bn.weight.mul_(self.output_mask)
        self.bn.bias.mul_(self.output_mask)


class OldPruneTool:
    '''
    can not prune downsample(only prune first two conv-bn)
    '''
    def __init__(self, percentage, model=None, devices='cpu', block=BasicBlock, channel_limit=8):
        super().__init__()
        self.model = model
        self.model.eval()

        self.prune_keep_size_model = deepcopy(self.model)
        self.percentage = percentage
        self.devices = devices
        self.prune_block = block
        self.channel_limit = channel_limit

        self.model_dict = self.get_model_dict(self.prune_keep_size_model)

        self.bn_thrsh = None

        self.compute_mask()

    def get_model_dict(self, model) -> OrderedDict:
        model_list = [module for _, module in enumerate(model.modules()) if isinstance(module, self.prune_block)]
        model_dict = OrderedDict()
        for i, module in enumerate(model_list):
            cur_model_list = [(name, module) for (name, module) in module.named_modules() if not isinstance(module, self.prune_block)]
            index = 0
            model_dict[i] = list()
            while index < cur_model_list.__len__():
                if not isinstance(cur_model_list[index][1], nn.Conv2d) or (
                        cur_model_list[index][1].stride == (2, 2) and cur_model_list[index][1].kernel_size == (
                        1, 1)) or 'downsample' in cur_model_list[index][0]:
                    index += 1
                else:
                    assert index + 1 < cur_model_list.__len__() and isinstance(cur_model_list[index + 1][1], nn.BatchNorm2d)
                    struct = ConvBn(cur_model_list[index][1], cur_model_list[index + 1][1], index, index + 1, i)
                    model_dict[i].append(struct)
                    index += 2
        return model_dict

    def get_bn_thresh(self) -> None:
        weight_list = list()
        for cur_block in self.model_dict.values():
            for item in cur_block:
                weight_list.append(item.bn.weight.data.abs().clone())
        weight_list = torch.cat(weight_list, dim=0)
        weight_list = weight_list.sort(dim=0)[0]

        index = int(weight_list.size(0) * self.percentage)
        thresh = float(weight_list[index].item())
        self.bn_thrsh = thresh
        print('bn thresh is: {}'.format(self.bn_thrsh))

    def compute_mask(self, verbose=True) -> None:
        if self.bn_thrsh is None:
            self.get_bn_thresh()
        assert self.bn_thrsh is not None

        for block in self.model_dict.values():
            self.compute_mask_for_block(block, verbose)

    def compute_mask_for_block(self, block: list, verbose=True) -> None:
        input_mask = None
        output_mask = None

        for i, module in enumerate(block):
            o_channel = module.bn.weight.size()[0]
            if i == 0:
                input_mask = torch.ones(module.conv.weight.size(1)).to(self.devices)
            else:
                input_mask = output_mask

            bn_weight = module.bn.weight
            if i == len(block) - 1:
                output_mask = torch.ones(bn_weight.size(0)).to(self.devices)
            else:
                remain_channel = bn_weight.data.abs().ge(self.bn_thrsh).sum().item()
                real_remain_channel = remain_channel // self.channel_limit * self.channel_limit + (
                    0 if remain_channel % self.channel_limit == 0 else self.channel_limit)
                if real_remain_channel == bn_weight.size(0):
                    output_mask = torch.ones(bn_weight.size(0)).to(self.devices)
                else:
                    real_prune_channel = bn_weight.size(0) - real_remain_channel
                    clone_weight = bn_weight.data.clone().abs().sort()[0]
                    real_thresh = clone_weight[real_prune_channel - 1].item()
                    output_mask = (bn_weight.data.abs().gt(real_thresh)).float()

                if verbose:
                    print('raw channel: {}\tprune channel: {}\tprune rate(cur module): {:.2f}%'.format(o_channel,
                                                                                                       o_channel - real_remain_channel, (
                                                                                                                   1. - real_remain_channel / o_channel) * 100))

            module.input_mask = input_mask
            module.output_mask = output_mask

    def mask_model_for_prune(self):
        for _, block in self.model_dict.items():
            conv_bn_nums = len(block)

            for i in range(conv_bn_nums - 1):
                cur_conv_bn = block[i]
                cur_output_mask = cur_conv_bn.output_mask
                cur_conv_bn.bn.weight.data.mul_(cur_output_mask)
                activation = F.relu((1 - cur_output_mask) * cur_conv_bn.bn.bias.data)

                next_conv_bn = block[i + 1]
                conv_sum = next_conv_bn.conv.weight.data.sum(dim=(2, 3))  # [out, in]
                offset = conv_sum.matmul(activation.view(-1, 1)).reshape(-1)  # [out]
                next_conv_bn.bn.running_mean.data.sub_(offset)

                cur_conv_bn.bn.bias.data.mul_(cur_output_mask)

    def get_compact_model(self, verbose=False) -> nn.Module:
        compact_model = deepcopy(self.model)
        compact_model_dict = self.get_model_dict(compact_model)

        assert compact_model_dict.__len__() == self.model_dict.__len__()
        num_blocks = len(compact_model_dict)

        for i in range(num_blocks):
            compact_block = compact_model_dict[i]
            block = self.model_dict[i]
            assert compact_block.__len__() == block.__len__()
            num_conv_bn_block = len(compact_block)

            for j in range(num_conv_bn_block):
                compact_conv_bn = compact_block[j]
                loose_conv_bn = block[j]
                raw_size = compact_conv_bn.conv.weight.size()

                input_mask = loose_conv_bn.input_mask
                output_mask = loose_conv_bn.output_mask

                compact_conv_bn.bn.weight.data = loose_conv_bn.bn.weight.data[output_mask.bool()].clone()
                compact_conv_bn.bn.bias.data = loose_conv_bn.bn.bias.data[output_mask.bool()].clone()
                compact_conv_bn.bn.running_mean.data = loose_conv_bn.bn.running_mean.data[output_mask.bool()].clone()
                compact_conv_bn.bn.running_var.data = loose_conv_bn.bn.running_var.data[output_mask.bool()].clone()

                temp = loose_conv_bn.conv.weight.data[:, input_mask.bool(), ...].clone()
                compact_conv_bn.conv.weight.data = temp[output_mask.bool(), ...].clone()
                new_size = compact_conv_bn.conv.weight.size()
                if verbose:
                    print('weight transfer {} to {}'.format(raw_size, new_size))

        return compact_model

    def get_prune_model(self) -> nn.Module:
        return self.prune_keep_size_model

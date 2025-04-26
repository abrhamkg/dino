import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision.models.resnet import Bottleneck
import random

# URLs for pre-trained models
resnext50_32x4d_url = 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'
resnext101_32x8d_url = 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'

def load_dino_mugs(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # remove `encoder.` prefix induced by MAE
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")


def remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.` if present
        new_state_dict[name] = v
    return new_state_dict

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def random_sample(prob, sampling_num):
    batch_size, channels, h, w = prob.shape
    # print(torch.any(prob < 0))
    # print(torch.all(abs(prob) < 1e-8))
    # import pdb; pdb.set_trace()
    return torch.multinomial((prob.view(batch_size * channels, -1) + 1e-8), sampling_num, replacement=True)

dropout_layers = 1  # how many layers to apply InfoDrop to
finetune_wo_infodrop = False  # when finetuning without InfoDrop, turn this on

class Info_Dropout(nn.Module):
    def __init__(self, indim, outdim, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, if_pool=False, pool_kernel_size=2, pool_stride=None,
                 pool_padding=0, pool_dilation=1):
        super(Info_Dropout, self).__init__()
        if groups != 1:
            raise ValueError('InfoDropout only supports groups=1')

        self.indim = indim
        self.outdim = outdim
        self.if_pool = if_pool
        self.drop_rate = 1.5
        self.temperature = 0.03
        self.band_width = 1.0
        self.radius = 3

        self.patch_sampling_num = 9

        self.all_one_conv_indim_wise = nn.Conv2d(self.patch_sampling_num, self.patch_sampling_num,
                                                 kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation,
                                                 groups=self.patch_sampling_num, bias=False)
        self.all_one_conv_indim_wise.weight.data = torch.ones_like(self.all_one_conv_indim_wise.weight, dtype=torch.float)
        self.all_one_conv_indim_wise.weight.requires_grad = False

        self.all_one_conv_radius_wise = nn.Conv2d(self.patch_sampling_num, outdim, kernel_size=1, padding=0, bias=False)
        self.all_one_conv_radius_wise.weight.data = torch.ones_like(self.all_one_conv_radius_wise.weight, dtype=torch.float)
        self.all_one_conv_radius_wise.weight.requires_grad = False


        if if_pool:
            self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride, pool_padding, pool_dilation)

        self.padder = nn.ConstantPad2d((padding + self.radius, padding + self.radius + 1,
                                         padding + self.radius, padding + self.radius + 1), 0)

    def initialize_parameters(self):
        self.all_one_conv_indim_wise.weight.data = torch.ones_like(self.all_one_conv_indim_wise.weight, dtype=torch.float)
        self.all_one_conv_indim_wise.weight.requires_grad = False

        self.all_one_conv_radius_wise.weight.data = torch.ones_like(self.all_one_conv_radius_wise.weight, dtype=torch.float)
        self.all_one_conv_radius_wise.weight.requires_grad = False


    def forward(self, x_old, x):
        if finetune_wo_infodrop:
            return x

        with torch.no_grad():
            distances = []
            padded_x_old = self.padder(x_old)
            sampled_i = torch.randint(-self.radius, self.radius + 1, size=(self.patch_sampling_num,)).tolist()
            sampled_j = torch.randint(-self.radius, self.radius + 1, size=(self.patch_sampling_num,)).tolist()
            for i, j in zip(sampled_i, sampled_j):
                tmp = padded_x_old[:, :, self.radius: -self.radius - 1, self.radius: -self.radius - 1] - \
                      padded_x_old[:, :, self.radius + i: -self.radius - 1 + i,
                      self.radius + j: -self.radius - 1 + j]
                distances.append(tmp.clone())
            distance = torch.cat(distances, dim=1)
            batch_size, _, h_dis, w_dis = distance.shape
            distance = (distance**2).view(-1, self.indim, h_dis, w_dis).sum(dim=1).view(batch_size, -1, h_dis, w_dis)
            distance = self.all_one_conv_indim_wise(distance)
            distance = torch.exp(
                -distance / distance.mean() / 2 / self.band_width ** 2)  # using mean of distance to normalize
            prob = (self.all_one_conv_radius_wise(distance) / self.patch_sampling_num) ** (1 / self.temperature)

            if self.if_pool:
                prob = -self.pool(-prob)  # min pooling of probability
            prob /= (prob.sum(dim=(-2, -1), keepdim=True) + 1e-8)


            batch_size, channels, h, w = x.shape
            try:
                random_choice = random_sample(prob, sampling_num=int(self.drop_rate * h * w))
            except Exception as e:
                print(e)
                print(prob)
                import pdb; pdb.set_trace()

            random_mask = torch.ones((batch_size * channels, h * w), device=x.device)
            random_mask[torch.arange(batch_size * channels, device=x.device).view(-1, 1), random_choice] = 0

        return x * random_mask.view(x.shape)


class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt Bottleneck Block with InfoDrop support
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=32,
                 base_width=4, dilation=1, norm_layer=None, if_dropout=False):
        super(ResNeXtBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = int(planes * (base_width / 64.)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.if_dropout = if_dropout
        if if_dropout:
            self.info_dropout1 = Info_Dropout(inplanes, width, kernel_size=3, stride=stride,
                                              padding=1, groups=1, dilation=1)
            self.info_dropout2 = Info_Dropout(width, width, kernel_size=3, stride=1,
                                              padding=1, groups=1, dilation=1)

    def forward(self, x):
        identity = x

        x_old = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.if_dropout:
            out = self.info_dropout1(x_old, out)

        x_old = out.clone()
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.if_dropout:
            out = self.info_dropout2(x_old, out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self, block, layers, jigsaw_classes=1000, classes=1000, 
                 groups=32, width_per_group=4, 
                 zero_init_residual=False,
                 replace_stride_with_dilation=None):
        super(ResNeXt, self).__init__()
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        
        norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.info_dropout = Info_Dropout(3, 64, kernel_size=7, stride=2, padding=3, if_pool=True,
                                         pool_kernel_size=3, pool_stride=2, pool_padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], if_dropout=(dropout_layers>=1))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, 
                                      dilate=replace_stride_with_dilation[0], 
                                      if_dropout=(dropout_layers>=2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, 
                                      dilate=replace_stride_with_dilation[1], 
                                      if_dropout=(dropout_layers>=3))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, 
                                      dilate=replace_stride_with_dilation[2], 
                                      if_dropout=(dropout_layers>=4))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, classes)
        # nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        # nn.init.constant_(self.fc.bias, 0)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNeXtBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
        
        # Initialize InfoDrop parameters
        for m in self.modules():
            if isinstance(m, Info_Dropout):
                print(m.drop_rate, m.temperature, m.band_width, m.radius)
                m.initialize_parameters()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, if_dropout=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, norm_layer, if_dropout=if_dropout
        ))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer, if_dropout=if_dropout
            ))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def get_last_layer_params(self):
        return list(self.class_classifier.parameters()) + list(self.jigsaw_classifier.parameters())

    def _forward_impl(self, x, if_get_feature=False):
        x_old = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if dropout_layers > 0:
            x = self.info_dropout(x_old, x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
    
    def forward(self, x, if_get_feature=False, **kwargs):
        return self._forward_impl(x, if_get_feature=if_get_feature)


def resnext50_32x4d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model with InfoDrop.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], 
                   groups=32, width_per_group=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnext50_32x4d_url), strict=False)
    print('ResNeXt-50 32x4d with InfoDrop loaded')
    return model


def resnext101_32x8d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model with InfoDrop.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], 
                   groups=32, width_per_group=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnext101_32x8d_url), strict=False)
    print('ResNeXt-101 32x8d with InfoDrop loaded')
    return model

if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    model_name = "dino_sfp_resnext50"

    model = resnext50_32x4d(pretrained=False)
    checkpoint = hf_hub_download(repo_id="eminorhan/"+model_name, filename=model_name+".pth")
    # print(checkpoint)
    # import pdb; pdb.set_trace()

    load_dino_mugs(model, checkpoint, "teacher")
    # print(model)

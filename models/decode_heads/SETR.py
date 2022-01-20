'''
mostly of the code was taken in https://github.com/gupta-abhay/setr-pytorch/blob/main/setr/SETR.py
'''
import torch.nn as nn 
import torch
import torch.nn.functional as F

class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs


class SETR_Naive(nn.Module):
    def __init__(self, embedding_dim, patch_dim, img_dim, num_classes):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.img_dim = img_dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1),),
        )
        self.bn1 = nn.BatchNorm2d(self.embedding_dim)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1),),
        )
        self.upsample = nn.Upsample(
            scale_factor=self.patch_dim, mode='bilinear'
        )

    def forward(self, x):
        x = self._reshape_output(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x
    
    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)
    


class SETR_PUP(nn.Module):
    def __init__(self, embedding_dim, patch_dim, img_dim, num_classes):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.img_dim = img_dim
        self.num_classes = num_classes

        extra_in_channels = int(self.embedding_dim / 4)
        in_channels = [
            self.embedding_dim,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
        ]
        out_channels = [
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            self.num_classes,
        ]

        modules = []
        for i, (in_channel, out_channel) in enumerate(
            zip(in_channels, out_channels)
        ):
            if self.patch_dim == 8 and i==1:
                continue
            else:
                modules.append(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=1,
                        stride=1,
                        padding=self._get_padding('VALID', (1, 1),),
                    )
                )
            if i != 4:
                modules.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decode_net = IntermediateSequential(
            *modules, return_intermediate=False
        )


    def forward(self, x):
        x = self._reshape_output(x)
        x = self.decode_net(x)
        return x
    
    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)


class SETR_MLA(nn.Module):
    def __init__(self, embedding_dim, patch_dim, img_dim, num_classes):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.img_dim = img_dim
        self.num_classes = num_classes

        self.net1_in, self.net1_intmd, self.net1_out = self._define_agg_net()
        self.net2_in, self.net2_intmd, self.net2_out = self._define_agg_net()
        self.net3_in, self.net3_intmd, self.net3_out = self._define_agg_net()
        self.net4_in, self.net4_intmd, self.net4_out = self._define_agg_net()

        # fmt: off
        self.output_net = IntermediateSequential(return_intermediate=False)
        self.output_net.add_module(
            "conv_1",
            nn.Conv2d(
                in_channels=self.embedding_dim, out_channels=self.num_classes,
                kernel_size=1, stride=1,
                padding=self._get_padding('VALID', (1, 1),),
            )
        )
        self.output_net.add_module(
            "upsample_1",
            nn.Upsample(scale_factor=4, mode='bilinear')
        )
        # fmt: on

    def forward(self, x, intmd_x, intmd_layers=None):
        assert intmd_layers is not None, "pass the intermediate layers for MLA"

        encoder_outputs = {}
        all_keys = []
        for i in intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()

        temp_x = encoder_outputs[all_keys[0]]
        temp_x = self._reshape_output(temp_x)
        key0_intmd_in = self.net1_in(temp_x)
        key0_out = self.net1_out(key0_intmd_in)

        temp_x = encoder_outputs[all_keys[1]]
        temp_x = self._reshape_output(temp_x)
        key1_in = self.net2_in(temp_x)
        key1_intmd_in = key1_in + key0_intmd_in
        key1_intmd_out = self.net2_intmd(key1_intmd_in)
        key1_out = self.net2_out(key1_intmd_out)

        temp_x = encoder_outputs[all_keys[2]]
        temp_x = self._reshape_output(temp_x)
        key2_in = self.net3_in(temp_x)
        key2_intmd_in = key2_in + key1_intmd_in
        key2_intmd_out = self.net3_intmd(key2_intmd_in)
        key2_out = self.net3_out(key2_intmd_out)

        temp_x = encoder_outputs[all_keys[3]]
        temp_x = self._reshape_output(temp_x)
        key3_in = self.net4_in(temp_x)
        key3_intmd_in = key3_in + key2_intmd_in
        key3_intmd_out = self.net4_intmd(key3_intmd_in)
        key3_out = self.net4_out(key3_intmd_out)

        out = torch.cat((key0_out, key1_out, key2_out, key3_out), dim=1)
        out = self.output_net(out)
        return out

    # fmt: off
    def _define_agg_net(self):
        model_in = IntermediateSequential(return_intermediate=False)
        model_in.add_module(
            "layer_1",
            nn.Conv2d(
                self.embedding_dim, int(self.embedding_dim / 2), 1, 1,
                padding=self._get_padding('VALID', (1, 1),),
            ),
        )

        model_intmd = IntermediateSequential(return_intermediate=False)
        model_intmd.add_module(
            "layer_intmd",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 2), 3, 1,
                padding=self._get_padding('SAME', (3, 3),),
            ),
        )

        model_out = IntermediateSequential(return_intermediate=False)
        model_out.add_module(
            "layer_2",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 2), 3, 1,
                padding=self._get_padding('SAME', (3, 3),),
            ),
        )
        model_out.add_module(
            "layer_3",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 4), 3, 1,
                padding=self._get_padding('SAME', (3, 3),),
            ),
        )
        model_out.add_module(
            "upsample", nn.Upsample(scale_factor=4, mode='bilinear')
        )
        return model_in, model_intmd, model_out
    # fmt: on
    
    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

from mmcv.cnn import build_norm_layer


class MLAHead(nn.Module):
    '''
    https://github.com/fudan-zvg/SETR/blob/main/mmseg/models/decode_heads/vit_mla_head.py
    '''
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        # head2 = self.head2(mla_p2)
        head2 = F.interpolate(self.head2(
            mla_p2), 4*mla_p2.shape[-1], mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head3(
            mla_p3), 4*mla_p3.shape[-1], mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head4(
            mla_p4), 4*mla_p4.shape[-1], mode='bilinear', align_corners=True)
        head5 = F.interpolate(self.head5(
            mla_p5), 4*mla_p5.shape[-1], mode='bilinear', align_corners=True)
        return torch.cat([head2, head3, head4, head5], dim=1)
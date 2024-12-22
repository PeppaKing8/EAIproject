#@markdown ### **Network**
#@markdown
#@markdown Defines a 1D UNet architecture `ConditionalUnet1D`
#@markdown as the noies prediction network
#@markdown
#@markdown Components
#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

"""
Note: This is copied from the colab notebook.
The main difference with the github repo code is in `class ConditionalUnet1D` -- this version makes some simplifications.
"""


from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusion_policy.resnet import resnet_custom_rgb, resnet_custom_rgbd
import os

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
    
class ImagePreprocessRGB(nn.Module):
    def __init__(self, output_channels=84):
        super().__init__()
        self.output_channels = output_channels
        # self.fc = nn.Linear(128*128*3, output_channels)
        self.resnet = resnet_custom_rgb()

    def forward(self, x):
        assert x.shape[-1] == 2 * 128 * 128 * 3
        x = x.view(x.shape[0] * 2, 128, 128, 3).permute(0, 3, 1, 2)
        x = self.resnet(x)
        x = x.view(-1, self.output_channels)
        return x
    
class ImagePreprocessRGBD(nn.Module):
    def __init__(self, output_channels=84):
        super().__init__()
        self.output_channels = output_channels
        ###################
        self.split_depth = False # choose the model type here
        ###################
        if self.split_depth:
            self.resnet_rgb = resnet_custom_rgb()
            self.split: int = 8
            self.thres_l = 0.01
            self.thres_r = 0.05
            self.resnet_depth = []
            self.threses = []
            
            length = (self.thres_r - self.thres_l) / self.split
            for i in range(self.split):
                self.resnet_depth.append(resnet_custom_rgbd(in_channels=1).to('cuda:0'))
                self.threses.append((self.thres_l + length * i, self.thres_l + length * (i + 1)))
            
            # self.resnet_depth = resnet_custom_rgbd(in_channels=1)
            self.fc = nn.Linear(output_channels * (self.split + 1), output_channels)
        else:
            self.resnet_rgb = resnet_custom_rgb()
            self.resnet_depth = resnet_custom_rgbd(in_channels=1)
            self.fc = nn.Linear(output_channels * 2, output_channels)

    def forward(self, x):
        assert x.shape[-1] == 2 * 128 * 128 * 4
        x = x.view(x.shape[0] * 2, 128, 128, 4).permute(0, 3, 1, 2) # (B, 4, 128, 128)
        if self.split_depth:
            rgb = x[:, :3, ...]
            depth = x[:, 3:, ...]
            
            out = self.resnet_rgb(rgb)
            out = out.view(-1, self.output_channels)
            
            # depth_discrete = torch.zeros_like(depth)
            for i in range(self.split):
                mask = (depth > self.threses[i][0]) & (depth <= self.threses[i][1])
                depth_map = depth * mask
                depth_map = (depth_map - self.threses[i][0]) / (self.threses[i][1] - self.threses[i][0])
                # depth_map = mask.to(torch.float32)
                depth_map = depth_map.to('cuda:0')
                depth_map.requires_grad = True
                out_depth = self.resnet_depth[i](depth_map)
                out_depth = out_depth.view(-1, self.output_channels)
                out = torch.cat([out, out_depth], dim=-1)
                    
            # depth.requires_grad = True
            # depth_exp_img = depth[0][0]
            # import matplotlib.pyplot as plt
            # plt.imshow(depth_exp_img.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            # plt.colorbar()
            # plt.show()
            # if not os.path.exists('output_images'):
            #     os.makedirs('output_images')
            # plt.savefig('output_images/depth_image.png')
            # rgb_exp_img = rgb[0].permute(1, 2, 0)
            # # # rgb_exp_img[:,:,2] = rgb_exp_img[:,:,1] = 0.0
            # plt.imshow(rgb_exp_img.cpu().numpy())
            # plt.savefig('output_images/rgb_image.png')
            # assert False
            
            # out_depth = self.resnet_depth(depth)
            # out_depth = out_depth.view(-1, self.output_channels)
            # out = torch.cat([out, out_depth], dim=-1)
            
            out = self.fc(out)
            return out
        else:
            rgb = x[:, :3, ...]
            depth = x[:, 3:, ...]
            out_rgb = self.resnet_rgb(rgb)
            out_depth = self.resnet_depth(depth)
            # print("image/", out_rgb.shape, out_depth.shape, self.output_channels)
            out_rgb = out_rgb.view(-1, self.output_channels)
            out_depth = out_depth.view(-1, self.output_channels)
            out = torch.cat([out_rgb, out_depth], dim=-1)
            out = self.fc(out)
            return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, args,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        # cond_dim = dsed + global_cond_dim # 148 = 64 + 84
        cond_dim = 148
        self.cond_dim = cond_dim
        # print("cond:", cond_dim, "=", dsed, "+", global_cond_dim)

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        
        if args.method == 'rgb':
            self.image_preprocess = ImagePreprocessRGB()
            self.method = 'rgb'
        elif args.method == 'rgbd':
            self.image_preprocess = ImagePreprocessRGBD()
            self.method = 'rgbd'
        elif args.method == 'state':
            self.method = 'state'
        else:
            raise ValueError(f"Invalid method: {args.method}")

        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M")

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            if self.method == 'rgb':
                assert global_cond.shape[-1] == 98304, f"global_cond shape: {global_cond.shape}"
                global_cond = self.image_preprocess(global_cond)
            elif self.method == 'rgbd':
                assert global_cond.shape[-1] == 131072, f"global_cond shape: {global_cond.shape}"
                if isinstance(global_cond, torch.cuda.ShortTensor):
                    global_cond = global_cond.to(torch.float32)
                global_cond = self.image_preprocess(global_cond)
            
            assert global_cond.shape[0] == global_feature.shape[0], f"global_cond shape: {global_cond.shape}, global_feature shape: {global_feature.shape}"
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
            assert global_feature.shape[-1] == self.cond_dim, f"global_feature shape: {global_feature.shape}; cond_dim: {self.cond_dim}"

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x

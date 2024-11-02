import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from PIL import Image
from torchvision import transforms

from einops import rearrange

from .mobius_utils import *

def angles_from_pixel_coords(coord):
    '''map from pixel coordinates (-1, 1) x (-1, 1) to 
    (-pi, pi) x (-pi/2, pi/2) rectangle'''

    out = torch.zeros_like(coord)
    out[:, :, 0] = coord[:, :, 1] * math.pi  + math.pi
    out[:, :, 1] = coord[:, :, 0] * 0.5 * math.pi
    return out

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def rotation_points(coord_, beta):
    coord = coord_.clone()
    beta = torch.tensor(beta) * (math.pi / 2)
    beta = beta.cuda()
    coord[:, :, :, 0] = coord[:, :, :, 0] * math.pi / 2 + math.pi / 2
    coord[:, :, :, 1] = coord[:, :, :, 1] * math.pi
    
    new_theta = torch.arccos(-torch.sin(beta)*torch.sin(coord[:, :, :, 0])*torch.cos(coord[:, :, :, 1])+torch.cos(beta)*torch.cos(coord[:, :, :, 0]))
    new_phi = torch.arctan2(torch.sin(coord[:, :, :, 0])*torch.sin(coord[:, :, :, 1]),
                                  torch.cos(beta)*torch.sin(coord[:, :, :, 0])*torch.cos(coord[:, :, :, 1])+torch.sin(beta)*torch.cos(coord[:, :, :, 0]))
    new_coord = torch.zeros_like(coord)
    new_coord[:, :, :, 0] = (new_theta - (math.pi / 2)) / (math.pi / 2)
    new_coord[:, :, :, 1] = new_phi / math.pi
    
    return new_coord

def rotation_map(coord_, im, window_size):
    '''coord_: [B, nH, nW, window ** 2 + 1, 2], im: [B, H, W, C]'''
    B, nH, nW = coord_.shape[:3]
    pos = coord_.clone()
    pos = pos.view(B, nH, nW, window_size, window_size, 2)
    pos = pos.permute(0, 1, 3, 2, 4, 5).reshape(B, nH * window_size, nW * window_size, 2)
    new_im = F.grid_sample(
                    im.permute(0, 3, 1, 2), pos.flip(-1),
                    mode='nearest', align_corners=False) # [B, C, H, W]

    return new_im

def rotation_process(H, W, window_size, x, shift=False):
    '''x: [B, C, H, W]'''
    # if H == 128:
    #     im_root = '173.jpg'
    #     im = transforms.ToTensor()(Image.open(im_root).convert('RGB'))
    #     x = im.unsqueeze(0).cuda()
    #     x = x.permute(0, 2, 3, 1)
    
    B = x.shape[0]
    nH = H // window_size
    nW = W // window_size

    template = make_coord([window_size, window_size])
    template[:, 0] = template[:, 0] / nH
    template[:, 1] = template[:, 1] / nW # [window ** 2, 2]
    template = template.view(1, 1, 1, template.shape[0], template.shape[1]).repeat(B, nH, nW, 1, 1) # [B, nH, nW, window ** 2, 2]
    template = template.cuda()

    lookup_table = torch.zeros([B, nH, nW, window_size ** 2, 2])
    lookup_table = lookup_table.cuda()

    for i in range(nH):
        latitude = 2 * (i + 0.5) / nH - 1
        rotated_window_points = rotation_points(template[:, i], latitude)
        # if x.shape[1] == 128:
        #     rotated_window_points = 2 * (rotated_window_points - latitude) + latitude
        lookup_table[:, i] = rotated_window_points

    for j in range(nW):
        longtitude = (j + 0.5) / nW
        lookup_table[:, :, j, :, 1] -= 2 * (1/2 - longtitude)
    
    lookup_table[..., 1][lookup_table[..., 1] > 1] -= 2
    lookup_table[..., 1][lookup_table[..., 1] < -1] += 2
    lookup_table[..., 0][lookup_table[..., 0] > 1] = 1
    lookup_table[..., 0][lookup_table[..., 0] < -1] = -1

    v = rotation_map(lookup_table, x, window_size)
  
    # if v.shape[-1] == 128:
    #     # from torchvision import transforms
    #     transforms.ToPILImage()(x[0, :, :, 1].detach().cpu()).save('feature_raw.jpg')
    #     transforms.ToPILImage()(v[0, 1].detach().cpu()).save('feature_rotation.jpg')
    #     assert False
    v = v.permute(0, 2, 3, 1)

    return v

def deform_process(H, W, window_size, x, scale=1, shift=False):
    '''scale: [B, 2, H, W]'''
    
    B = x.shape[0]
    nH = H // window_size
    nW = W // window_size

    template = make_coord([H, W])
    template = template.unsqueeze(0).repeat(B, 1, 1).view(B, nH, window_size, nW, window_size, 2).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, nH, nW, -1, 2) # [B, nH, nW, window ** 2, 2]
    template = template.cuda()

    scale = scale.view(B, 2, nH, window_size, nW, window_size)
    scale = scale.permute(0, 2, 4, 3, 5, 1).contiguous().view(B, nH, nW, -1, 2)
   
    template = (template + scale).clamp_(-1, 1)
    
    template[..., 1][template[..., 1] > 1] -= 2
    template[..., 1][template[..., 1] < -1] += 2
    template[..., 0][template[..., 0] > 1] = 0.999
    template[..., 0][template[..., 0] < -1] = -0.999

    v = rotation_map(template, x, window_size)
    # print(x.shape, v.shape)
  
    # if v.shape[-1] == 256:
    #     # from torchvision import transforms
    #     transforms.ToPILImage()(x[0, :, :, 1].detach().cpu()).save('feature_raw.jpg')
    #     transforms.ToPILImage()(v[0, 1].detach().cpu()).save('feature_rotation.jpg')
    #     assert False
    v = v.permute(0, 2, 3, 1)

    return v
    
class LayerNorm(nn.Module):
    r""" Copied from ConvNeXt, LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.rpe_table = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.lepe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, y, mask, B, H, W):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # [1]: x,y,x, relative [2]: x,y,x, no relative [3]: y,x,x, relative [4]: y,x,x, no relative
        q = self.q(x)
        q_windows = q.view(-1, self.window_size[0], self.window_size[0], self.dim)
        q_windows = window_reverse(q_windows, self.window_size[0], H, W)  # B H' W' C
        q_windows = q_windows.permute(0, 3, 1, 2)
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k = self.k(y).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B_, N, C)
        # q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
      
        residual_windows = self.rpe_table(q_windows).permute(0, 2, 3, 1)
        lepe_windows = window_partition(residual_windows, self.window_size[0])  # nW*B, window_size, window_size, C
        lepe_windows = lepe_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Process relative positional encoding
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        
        # assert self.dim % v.shape[-1] == 0, "self.dim % v.shape[-1] != 0"
        # repeat_num = self.dim // v.shape[-1]
        # v = v.view(B_, N, self.num_heads // repeat_num, -1).transpose(1, 2).repeat(1, repeat_num, 1, 1)

        assert self.dim == v.shape[-1], "self.dim != v.shape[-1]"
        v = v.view(B_, N, self.num_heads, -1).transpose(1, 2)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        x = x + lepe_windows

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PanoSA(nn.Module):
    """ 
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, rotation=True, localconv=True, interact=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

        # CPE
        self.pos = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        # self.abs_encoder = nn.Linear(5, dim)

        self.rotation = rotation
        self.localconv = localconv
        self.interact = interact

        # Dilate template
        # self.offsets = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=1, stride=1),
        #     LayerNorm(dim, data_format="channels_first"),
        #     nn.GELU(),
        #     nn.Conv2d(dim, 2, kernel_size=1, stride=1)
        # )

        # self.planar = Planar_Interaction(dim=dim)
        # self.interaction = Feat_Interaction(dim=dim)
        # self.conv_interaction = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, groups=dim)

    def _pano_abs_position(self, x_bchw):
        """
        Obtain pano-style absolute positional embeddings
        @param x_bchw: (batch_size, channel, height, width), input feature map
        @return: a two-element tuple
            1. encoded_1chw (1, channel, height, width): absolute encodings,
            2. uv_12hw (1, 2, height, width): uv coordinates, u: [-pi, pi], v: [-0.5pi, pi]
        """
        B, C, H, W = x_bchw.shape
        uv = make_coord([H, W], flatten=False)
        uv = uv.cuda()
        uv_angle = angles_from_pixel_coords(uv)
        # uv_hw2 = make_uv_hw2(H, W, device=device)
    
        xyz_coord = torch.stack([
            torch.cos(uv_angle[..., 1]) * torch.cos(uv_angle[..., 0]),
            torch.cos(uv_angle[..., 1]) * torch.sin(uv_angle[..., 0]),
            torch.sin(uv_angle[..., 1]),
        ], -1)

        uvxyz = torch.cat([xyz_coord, uv_angle], -1)

        encoded_1hwc = self.abs_encoder(uvxyz[None]) # xyzyx_coord
        encoded_1chw = encoded_1hwc.permute(0, 3, 1, 2)
        uv_angle = uv_angle.permute(2, 0, 1)
        uv_angle = uv_angle[None, ...]
        return encoded_1chw, uv_angle

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        # uvxyz_pos, _ = self._pano_abs_position(x.permute(0, 2, 1).view(B, C, H, W))

        # absolute positional encoding
        x = x + self.pos(x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).flatten(1, 2)
        # uvxyz_pos = uvxyz_pos.repeat(B, 1, 1, 1).view(B, C, L).permute(0, 2, 1)
        # x = x + uvxyz_pos

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # deform_offsets = self.offsets(x.permute(0, 3, 1, 2)).contiguous()       #B, 2, h // window_size, w // window_size
        # offset_range = torch.tensor([1.0 / (H - 1.0), 1.0 / (W - 1.0)]).cuda().contiguous().reshape(1, 2, 1, 1)
        # deform_offsets = deform_offsets.tanh().mul(offset_range).mul(2.)
        
        if self.rotation:
            v = rotation_process(H, W, self.window_size, x, shift=False)
            # v = deform_process(H, W, self.window_size, x, scale=deform_offsets, shift=False)
            # theta = 0
            # phi = np.pi / 2
            # M_scale = np.array([[1, 0], [0, 1]])
            # M_horizon = np.array([[np.cos(theta) + 1j * np.sin(theta), 0], [0, 1]])
            # M_vertical = np.array([[np.cos(phi / 2), np.sin(phi / 2)], [-np.sin(phi / 2), np.cos(phi / 2)]])
            # M = M_horizon @ M_vertical @ M_scale
            # M = torch.from_numpy(M).cuda()
            # coord_hr = make_coord([H, W], flatten=True).unsqueeze(0)
            # v = warp_mobius_image(x.view(B, H * W, C), M, coord_hr.cuda(), pole='North')
            # v = v.view(B, H, W, C)
        else: 
            v = x

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_v = torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            shifted_v = v
            attn_mask = None
   
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(shifted_v, self.window_size)  # nW*B, window_size, window_size, C
        v_windows = v_windows.view(-1, self.window_size * self.window_size, v_windows.shape[-1])  # nW*B, window_size*window_size, C
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, v_windows, attn_mask, B, H, W)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PanoBlock(nn.Module):
    """ A basic Block for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 rotation=True,
                 localconv=True,
                 interact=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        # self.blocks = nn.ModuleList([
        #     PanoSA(
        #         dim=dim,
        #         num_heads=num_heads,
        #         window_size=window_size,
        #         shift_size=0 if (i % 2 == 0) else window_size // 2,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop,
        #         attn_drop=attn_drop,
        #         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #         norm_layer=norm_layer,
        #         rotation=rotation,
        #         localconv=localconv,
        #         interact=interact)
        #     for i in range(depth)])
        self.blocks = nn.ModuleList([
            PanoSA(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                rotation=rotation,
                localconv=localconv,
                interact=interact)
            for i in range(depth)])

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        img_mask = torch.zeros((1, H, W, 1)).cuda()  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        
        return x, H, W


class PanoLayer(nn.Module):
    def __init__(self,
                 embed_dim=96,
                 window_size=7,
                 num_heads=4,
                 depth=2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 rotation=True,
                 localconv=True,
                 interact=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        self.pano_block = PanoBlock(
                dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=False,
                rotation=rotation,
                localconv=localconv,
                interact=interact)
        
        self.window_size = window_size

        layer = norm_layer(embed_dim)
        layer_name = 'norm'
        self.add_module(layer_name, layer)


    def forward(self, x):
        '''x: [B,C,H,W]'''
        Wh, Ww = x.size(2), x.size(3)
        
        x = x.flatten(2).transpose(1, 2) # [B, H*W, C]
        
        x_out, H, W = self.pano_block(x, Wh, Ww)
       
        norm_layer = getattr(self, f'norm')
        x_out = norm_layer(x_out)
        out = x_out.view(-1, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        # if out.shape[2] == 128:
        #     transforms.ToPILImage()(out[0, 0].detach().cpu()).save('feature_out.jpg')
        #     # transforms.ToPILImage()(local_out[0, :, 0].view(H, W).detach().cpu()).save('feature_local.jpg')
        #     assert False

        return out
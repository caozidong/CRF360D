import os

import numpy as np
import random
import math
from math import pi

import torch
import torch.nn.functional as F

def warp_mobius_coord(x, M, coord, get_sphere=False, pole='North'):
    '''Input: x->tensor:(b,n,c)
              M->tensor:(2,2)
              coord->tensor:(b,n,2)
       Output: coord->tensor:(b,2,n)
    '''
    # coord = make_coord([h, w], flatten=True)
    b, n, c = x.shape
    M_inv = torch.linalg.inv(M).unsqueeze(0)
    
    coord_in = coord
    coord_in_a = angles_from_pixel_coords(coord_in)
    coord_in_s = sphere_from_angles(coord_in_a, b, n)
    coord_in_c = CP1_from_sphere(coord_in_s, b, n, pole)
    coord_out_c = torch.matmul(M_inv, coord_in_c.permute(0, 2, 1)).permute(0, 2, 1)
    coord_out_s = sphere_from_CP1(coord_out_c, b, n, pole)
    coord_out_a = angles_from_sphere(coord_out_s, b, n)
    coord_out = pixel_coords_from_angles(coord_out_a)

    if get_sphere:
        return coord_in_s, coord_out_s, coord_out
    else:
        return coord_out

def warp_mobius_image(x, M, coord, pole='North'):
    '''Input: x->tensor:(b,n,c)
              M->tensor:(2,2)
              coord->tensor:(b,n,2)
       Output: coord->tensor:(b,2,n)
    '''
    
    coord_t = warp_mobius_coord(x, M, coord, get_sphere=False, pole=pole)

    #Convert from pixels to matrix, by assuming that H:W=1:2
    b, n, c = x.shape
    h = int(math.sqrt(n / 2))
    w = 2 * h
    x = x.permute(0, 2, 1).view(b, c, h, w)
    
    transform_im = F.grid_sample(
                x, coord_t.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :]
    transform_im = transform_im.permute(0, 2, 1)

    return transform_im

def angles_from_pixel_coords(coord):
    '''map from pixel coordinates (-1, 1) x (-1, 1) to 
    (0, 2*pi) x (-pi/2, pi/2) rectangle'''

    out = torch.zeros_like(coord)
    out[:, :, 0] = coord[:, :, 1] * pi +  pi    
    out[:, :, 1] = coord[:, :, 0] * 0.5 * pi
    return out

def sphere_from_angles(coord, b, n):
    '''
    inverse equirectangular projections, 
    ie. map from (0,2*pi) x (-pi/2,pi/2) rectangle to sphere in R^3
    Parameters:
    b: batch size
    n: Number of points
    '''
    out = torch.zeros([b, n, 3]).cuda()
    lon, lat = coord[:, :, 0], coord[:, :, 1]

    out[:, :, 0] = torch.cos(lat) * torch.cos(lon) # x
    out[:, :, 1] = torch.cos(lat) * torch.sin(lon) # y
    out[:, :, 2] = torch.sin(lat) # z
    return out
    
def CP1_from_sphere(coord, b, n, pole):
    """map from sphere in R^3 to CP^1"""
    out = torch.zeros([b, n, 2], dtype=torch.complex128).cuda()
    x, y, z = coord[:, :, 0], coord[:, :, 1], coord[:, :, 2]

    if pole == 'North':
        mask = z < 0
        out[:, :, 0] = torch.where(mask, x + 1j * y, 1 + z)
        out[:, :, 1] = torch.where(mask, 1 - z, x - 1j * y)
    elif pole == 'Equator':
        mask = x < 0
        out[:, :, 0] = torch.where(mask, y + 1j * z, 1 + x)
        out[:, :, 1] = torch.where(mask, 1 - x, y - 1j * z)
    else:
         assert False
    return out

def sphere_from_CP1(coord, b, n, pole):
    """map from CP^1 to sphere in R^3"""
    out = torch.zeros([b, n, 3]).cuda()
    z1, z2 = coord[:, :, 0], coord[:, :, 1]
    mask = torch.abs(z2) > torch.abs(z1)
    z = torch.where(mask, z1 / z2, torch.conj(z2 / z1))
    x, y = torch.real(z), torch.imag(z)

    if pole == 'North':
        denom = 1 + torch.pow(x, 2) + torch.pow(y, 2)
        out[:, :, 0] = 2 * x / denom
        out[:, :, 1] = 2 * y / denom
        out[:, :, 2] = (denom - 2) / denom * (2 * mask - 1)
    elif pole == 'Equator':
        denom = 1 + torch.pow(x, 2) + torch.pow(y, 2)
        out[:, :, 1] = 2 * x / denom
        out[:, :, 2] = 2 * y / denom
        out[:, :, 0] = (-2 + denom) / denom * (2 * mask - 1)
    else:
        assert False
    return out

def angles_from_sphere(coord, b, n):

    out = torch.zeros([b, n, 2]).cuda()
    x, y, z = coord[:, :, 0], coord[:, :, 1], coord[:, :, 2]

    out[:, :, 0] = torch.arctan2(y, x)
    out[:, :, 0] = out[:, :, 0] % (2 * pi)

    r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
    out[:, :, 1] = torch.arctan2(z, r)

    return out

def pixel_coords_from_angles(coord):
    out = torch.zeros_like(coord)

    out[:, :, 1] = coord[:, :, 0] / pi - 1
    out[:, :, 0] = coord[:, :, 1] / (0.5 * pi)
    return out

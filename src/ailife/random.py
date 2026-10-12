"""Counter-based streams: observers and births never advance world randomness."""
import torch


def hash32(x):
    x = x.to(torch.int64) & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    return (x ^ (x >> 16)) & 0xFFFFFFFF


def uniform(key, stream=0):
    return ((hash32(key + stream * 0x9E3779B1).to(torch.float64) + 0.5) / 4294967296.0).float()


def normal(key, stream=0):
    u = uniform(key, stream).clamp_min(1e-7)
    v = uniform(key, stream + 1)
    return torch.sqrt(-2 * torch.log(u)) * torch.cos(6.283185307179586 * v)

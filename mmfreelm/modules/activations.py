# -*- coding: utf-8 -*-

# Copyright (c) 2023, Tri Dao.

import torch
import torch.nn.functional as F


@torch.jit.script
def swish(x):
    return F.silu(x)

# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@torch.jit.script
def bias_gelu(y, bias):
    x = bias + y
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(dtype=y.dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def bias_gelu_bwd(g, y, bias):
    """Assume that y has shape (B, D) and bias has shape (D)"""
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    grad_y = ff * g
    return grad_y.to(dtype=y.dtype), grad_y.sum(dim=(0), dtype=bias.dtype)


class GeLUFunction(torch.autograd.Function):

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_bwd(grad_output, input, bias)
        return tmp, tmp


bias_gelu_impl = GeLUFunction.apply


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@torch.jit.script
def gelu_fwd(x):
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(dtype=x.dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def gelu_bwd(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    return (ff * g).to(dtype=x.dtype)


class FastGeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        tmp = gelu_bwd(grad_output, input)
        return tmp


fast_gelu_impl = FastGeLUFunction.apply


@torch.jit.script
def relu_bwd(g, x):
    return torch.where(x >= 0, g, 0.0).to(dtype=x.dtype)


@torch.jit.script
def sqrelu_fwd(x):
    r = F.relu(x)
    return (r * r).to(dtype=x.dtype)


@torch.jit.script
def swiglu_fwd(x, y):
    return F.silu(x) * y


@torch.jit.script
def swiglu_bwd(x, y, g, dx, dy):
    x_sigmoid = 1.0 / (1.0 + torch.exp(-x))
    dx = x_sigmoid * (1 + x * (1.0 - x_sigmoid)) * g * y
    dy = x * x_sigmoid * g


@torch.jit.script
def swiglu_bwd_with_output(x, y, g, dx, dy, z):
    x_sigmoid = 1.0 / (1.0 + torch.exp(-x))
    x_swish = swish(x)
    dx = x_sigmoid * (1 + x * (1.0 - x_sigmoid)) * g * y
    dy = x_swish * g
    z = x_swish * y


class SwiGLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return swiglu_fwd(x, y)

    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        return swiglu_bwd(x, y, dout)


class SwiGLULinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y, weight, bias):
        z = swiglu_fwd(x, y)
        out = F.linear(z.to(weight.dtype), weight, bias)
        # We don't store z, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(x, y, weight)
        ctx.linear_bias_is_none = bias is None
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        x, y, weight = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dz = F.linear(dout, weight.t()).view_as(x)
        dx, dy, z = swiglu_bwd_with_output(x, y, dz)
        dlinear_weight = torch.einsum("bo,bi->oi", dout, z.reshape(-1, z.shape[-1]))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        return dx, dy, dlinear_weight, dlinear_bias


swiglu = SwiGLUFunction.apply

swiglu_linear = SwiGLULinearFunction.apply

ACT2FN = {
    'silu': swish,
    'swish': swish,
    'gelu': fast_gelu_impl,
    'bias_gelu': bias_gelu_impl,
}

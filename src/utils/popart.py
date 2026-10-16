import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PopArt(nn.Module):
    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99, epsilon=1e-5, device=torch.device("cpu")):
        super(PopArt, self).__init__()

        self.beta = beta  # اصلاح مقدار Beta از 0.99999 به 0.99 برای تطبیق سریع‌تر با تغییرات بازده
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)
        
        self.register_buffer("stddev", torch.ones(output_shape, **self.tpdv))
        self.register_buffer("mean", torch.zeros(output_shape, **self.tpdv))
        self.register_buffer("mean_sq", torch.zeros(output_shape, **self.tpdv))
        self.register_buffer("debiasing_term", torch.tensor(0.0, **self.tpdv))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        return F.linear(input_vector, self.weight, self.bias)

    @torch.no_grad()
    def update(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv).detach()
        
        if input_vector.numel() == 0:
            return

        old_mean, old_stddev = self.mean.clone(), self.stddev.clone()

        flat_inputs = input_vector.reshape(-1, self.output_shape)
        batch_mean = flat_inputs.mean(dim=0)
        batch_sq_mean = (flat_inputs ** 2).mean(dim=0)

        self.mean.data.copy_(self.mean * self.beta + batch_mean * (1.0 - self.beta))
        self.mean_sq.data.copy_(self.mean_sq * self.beta + batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.data.copy_(self.debiasing_term * self.beta + 1.0 * (1.0 - self.beta))

        deb_mean, deb_var = self.debiased_mean_var()
        new_stddev = torch.sqrt(deb_var)

        self.weight.data.copy_(self.weight.data * (old_stddev / new_stddev).unsqueeze(1))
        self.bias.data.copy_((old_stddev * self.bias.data + old_mean - deb_mean) / new_stddev)
        
        self.stddev.data.copy_(new_stddev)

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-4)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.debiased_mean_var()
        return (input_vector - mean) / torch.sqrt(var)

    def denormalize(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.debiased_mean_var()
        return input_vector * torch.sqrt(var) + mean



class PopArt__(nn.Module):
    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)
        
        self.register_buffer("stddev", torch.ones(output_shape, **self.tpdv))
        self.register_buffer("mean", torch.zeros(output_shape, **self.tpdv))
        self.register_buffer("mean_sq", torch.zeros(output_shape, **self.tpdv))
        self.register_buffer("debiasing_term", torch.tensor(0.0, **self.tpdv))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        return F.linear(input_vector, self.weight, self.bias)

    @torch.no_grad()
    def update(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv).detach()
        
        # اصلاح: حفظ شکل ابعاد خروجی برای محاسبه صحیح میانگین و واریانس
        if input_vector.numel() == 0:
            return

        old_mean, old_stddev = self.mean.clone(), self.stddev.clone()

        # محاسبه میانگین روی تمام ابعاد به جز بعد خروجی (که بعد آخر یا اول است)
        flat_inputs = input_vector.reshape(-1, self.output_shape)
        batch_mean = flat_inputs.mean(dim=0)
        batch_sq_mean = (flat_inputs ** 2).mean(dim=0)

        self.mean.copy_(self.mean * self.beta + batch_mean * (1.0 - self.beta))
        self.mean_sq.copy_(self.mean_sq * self.beta + batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.copy_(self.debiasing_term * self.beta + 1.0 * (1.0 - self.beta))

        deb_mean, deb_var = self.debiased_mean_var()
        new_stddev = torch.sqrt(deb_var)

        # آپدیت وزنی دقیق PopArt
        self.weight.copy_(self.weight * (old_stddev / new_stddev).unsqueeze(1))
        self.bias.copy_((old_stddev * self.bias + old_mean - deb_mean) / new_stddev)
        
        self.stddev.copy_(new_stddev)
    
    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.debiased_mean_var()
        return (input_vector - mean) / torch.sqrt(var)

    def denormalize(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.debiased_mean_var()
        return input_vector * torch.sqrt(var) + mean


class PopArt_____(torch.nn.Module):
    def __init__(self, input_shape, output_shape, 
                 norm_axes=2, beta=0.99999, epsilon=1e-5, 
                 device=torch.device("cpu")):
        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)
        
        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        return F.linear(input_vector, self.weight, self.bias)
    
    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        
        old_mean, old_stddev = self.mean, self.stddev

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        self.stddev = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)

        self.weight = self.weight * old_stddev / self.stddev
        self.bias = (old_stddev * self.bias + old_mean - self.mean) / self.stddev

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.debiased_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        return out

    def denormalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        return out
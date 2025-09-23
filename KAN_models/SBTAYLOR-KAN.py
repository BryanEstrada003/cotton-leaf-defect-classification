"""
SBTAYLOR-KAN: Spline-based Taylor-KAN Model
=============================
Paper Title: "Taylor-Series Expanded Kolmogorov-Arnold Network for Medical Imaging Classification"

This module implements the **Spline-based Taylor-KAN model**, designed for medical image classification 
tasks such as Chest X-ray analysis. It integrates:

1. **KANLinear**:
   - Kolmogorov–Arnold inspired B-spline linear layer
   - Combines standard linear transform with spline-based nonlinearity

2. **TaylorSeriesApproximation**:
   - Truncated Taylor series expansion for functional priors
   - Approximates sin(x) and injects domain-inspired features

3. **Net** (Spline-based Taylor-KAN):
   - CNN feature extractor
   - Taylor functional transformation of features
   - KANLinear fully connected layers for classification

**Key Contributions:**
- Introduces spline-based Kolmogorov–Arnold linear layers for flexible function approximation
- Integrates Taylor series expansions to embed analytical priors
- Optimized for medical imaging tasks such as Chest X-ray classification



import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
#                  CUSTOM PYTORCH LAYERS
# =========================================================

class KANLinear(nn.Module):
    """
    Kolmogorov–Arnold inspired B-spline linear layer.
    Combines a standard linear transform with spline-based
    feature transformation for flexible representation.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True, base_activation=nn.SiLU,
                 grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Create grid for B-spline basis
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
                .expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        # Base and spline weights
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))

        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with noise for spline flexibility."""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.out_features, self.in_features, self.grid_size + self.spline_order) - 0.5)
                     * self.scale_noise / self.grid_size)
            grid_slice = self.grid.T[self.spline_order:self.spline_order + noise.size(2)]
            coeffs = self.curve2coeff(grid_slice, noise)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * coeffs
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """Compute B-spline basis for each input feature."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid.unsqueeze(0)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            denom1 = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
            denom2 = grid[:, :, (k + 1):] - grid[:, :, 1:-k]
            denom1 = torch.where(denom1 == 0, torch.ones_like(denom1), denom1)
            denom2 = torch.where(denom2 == 0, torch.ones_like(denom2), denom2)
            bases = ((x - grid[:, :, :-(k + 1)]) / denom1 * bases[:, :, :-1] +
                     (grid[:, :, (k + 1):] - x) / denom2 * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """Fit spline coefficients using least squares."""
        A = self.b_splines(x)  
        A = A.permute(1, 0, 2)
        B = y.permute(1, 0, 2)
        coeffs = torch.zeros((y.size(0), y.size(1), A.size(2)), device=x.device, dtype=x.dtype)
        for i in range(y.size(1)):
            A_i = A[i]
            B_i = B[i]
            for j in range(y.size(0)):
                solution = torch.linalg.lstsq(A_i, B_i[j].unsqueeze(1)).solution
                coeffs[j, i, :] = solution.squeeze(1)
        return coeffs.contiguous()

    @property
    def scaled_spline_weight(self):
        """Return spline weights scaled if standalone scaler is enabled."""
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor):
        """Forward pass with base + spline transformation."""
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_basis = self.b_splines(x).view(x.size(0), -1)
        spline_output = F.linear(spline_basis, self.scaled_spline_weight.view(self.out_features, -1))
        output = base_output + spline_output
        return output.view(*original_shape[:-1], self.out_features)


class TaylorSeriesApproximation(nn.Module):
    """Approximates sin(x) using truncated Taylor series."""
    def __init__(self, n_terms=5):
        super().__init__()
        factorials = [math.factorial(2 * n + 1) for n in range(n_terms)]
        self.register_buffer('factorials', torch.tensor(factorials, dtype=torch.float32))

    def forward(self, x):
        approximation = torch.zeros_like(x)
        for n, fact in enumerate(self.factorials):
            approximation += ((-1) ** n) * (x ** (2 * n + 1)) / fact
        return approximation

# =========================================================
#                  MAIN NETWORK
# =========================================================

class Net(nn.Module):
    """
    Spline-based Taylor-KAN Model for Medical Imaging Classification.
    Combines:
    1. CNN feature extractor
    2. Taylor series functional transformation
    3. KANLinear fully connected layers
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Functional transformation
        self.taylor_approx = TaylorSeriesApproximation(n_terms=5)

        # Fully connected layers (KANLinear)
        self.fc1_input_size = self._get_conv_output_size((3, 224, 224))
        self.fc1 = KANLinear(self.fc1_input_size, 120)
        self.fc2 = KANLinear(120, 84)
        self.fc3 = KANLinear(84, num_classes)

    def _get_conv_output_size(self, input_shape):
        """Compute flattened feature size after conv+pool layers."""
        dummy_input = torch.zeros(1, *input_shape)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(1, -1).shape[1]

    def forward(self, x):
        """Forward pass of the network."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.taylor_approx(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

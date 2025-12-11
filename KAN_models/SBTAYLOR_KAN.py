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

"""

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

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

        # Inicialización de los splines
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            # Usamos el método curve2coeff vectorizado
            try:
                self.spline_weight.data.copy_(
                    self.curve2coeff(
                        self.grid.T[self.spline_order : -self.spline_order], noise
                    )
                )
            except Exception as e:
                # Fallback a inicialización aleatoria si lstsq falla
                print(
                    f"Warning: curve2coeff falló durante la inicialización: {e}. Usando inicialización aleatoria para los splines."
                )
                nn.init.normal_(self.spline_weight, mean=0.0, std=self.scale_noise)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:-k])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        # x: (grid_size+1, in_features)
        # y: (grid_size+1, in_features, out_features)
        A = self.b_splines(x.permute(1, 0)).permute(1, 0, 2)
        B = y.permute(1, 2, 0)
        # Solución vectorizada de mínimos cuadrados
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    def forward(self, x: torch.Tensor):
        # Guardar forma original para la salida
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        # Salida de la capa base
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Salida de la capa de splines
        spline_basis = self.b_splines(x)
        scaled_spline_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        # Dentro de KANLinear.forward()
        spline_output = torch.einsum("bfi,ofi->bo", spline_basis, scaled_spline_weight)

        # Combinar salidas
        output = base_output + spline_output

        # Restaurar forma original
        output = output.view(*original_shape[:-1], self.out_features)
        return output


class TaylorSeriesApproximation(nn.Module):
    """Approximates sin(x) using truncated Taylor series."""

    def __init__(self, n_terms=5):
        super().__init__()
        factorials = [math.factorial(2 * n + 1) for n in range(n_terms)]
        self.register_buffer(
            "factorials", torch.tensor(factorials, dtype=torch.float32)
        )

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
        self.bn1 = nn.BatchNorm1d(self.fc1_input_size)
        self.fc1 = KANLinear(self.fc1_input_size, 120)
        self.dropout1 = nn.Dropout(p=0.25)  # <-- AÑADIR DROPOUT (50%)
        self.fc2 = KANLinear(120, 84)
        self.dropout2 = nn.Dropout(p=0.5)  # <-- AÑADIR DROPOUT
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
        x = self.bn1(x)
        x = torch.tanh(x)
        x = self.taylor_approx(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # <-- APLICAR DROPOUT
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # <-- APLICAR DROPOUT
        return self.fc3(x)

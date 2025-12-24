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
        # Inicialización de la parte lineal base (Kaiming es estándar)
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

        # Inicialización de los splines
        with torch.no_grad():
            # El ruido debe tener la misma forma que self.spline_weight:
            # (out_features, in_features, grid_size + spline_order)
            noise = (
                (
                    torch.rand(self.out_features, self.in_features, self.grid_size + self.spline_order) 
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            
            # 2. USO DE LA VARIABLE:
            # Copiamos el 'noise' calculado dentro de los pesos del spline
            self.spline_weight.data.copy_(noise)

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

    def __init__(self, num_classes=2, input_size=(3, 224, 224), dropout_1=0.5, dropout_2=0.5):
        super().__init__()
        
        # --- 1. FEATURE EXTRACTOR (Agrupado correctamente) ---
        # Usamos nn.Sequential para que 'self.features' exista y sea iterable/llamable
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 224 -> 112

            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 112 -> 56

            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 56 -> 28

            # Bloque 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 28 -> 14
        )

        # --- CABEZA KAN ---
        self.taylor_approx = TaylorSeriesApproximation(n_terms=5)

        # --- CÁLCULO AUTOMÁTICO DE DIMENSIONES ---
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            
            # Ahora esto SÍ funcionará porque self.features ya está definido arriba
            dummy_output = self.features(dummy_input)
            
            # Calculamos el tamaño aplanado
            self.flatten_dim = dummy_output.view(1, -1).size(1)
            print(f"Dimensiones calculadas automáticamente para FC1: {self.flatten_dim}")

        # Definición de capas lineales usando la dimensión calculada
        self.fc1 = KANLinear(self.flatten_dim, 256)
        self.dropout = nn.Dropout(dropout_1)
        self.fc2 = KANLinear(256, 128)
        self.dropout2 = nn.Dropout(dropout_2)
        self.fc3 = KANLinear(128, num_classes)

    def forward(self, x):
        # 1. Extraer características (Ahora es una sola llamada limpia)
        x = self.features(x)

        # 2. Aplanar
        x = torch.flatten(x, 1)

        # 3. Estabilización y Taylor
        x = torch.tanh(x) 
        x = self.taylor_approx(x)

        # 4. Clasificación KAN
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x) 
        
        return x

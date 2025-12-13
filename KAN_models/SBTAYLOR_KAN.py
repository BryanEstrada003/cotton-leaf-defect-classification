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

        # --- BLOQUE 1: Detecta bordes y colores simples ---
        # De 3 canales (RGB) a 32 filtros
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # --- BLOQUE 2: Detecta formas geométricas ---
        # De 32 a 64 filtros
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # --- BLOQUE 3: Detecta texturas complejas (Hongos/Manchas) ---
        # De 64 a 128 filtros
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # --- BLOQUE 4: Refinamiento ---
        # Mantenemos 128 pero profundizamos
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Capa de Pooling (reduce tamaño a la mitad)
        self.pool = nn.MaxPool2d(2, 2)

        # --- CÁLCULO AUTOMÁTICO DE DIMENSIONES ---
        # Esto evita que tengas que calcular "16*53*53" a mano.
        # Le pasamos una imagen falsa de 224x224 para ver cuánto mide al final.
        self._to_linear = None
        self._get_conv_output(
            (3, 224, 224)
        )  # <--- AJUSTA 224 AL TAMAÑO DE TUS IMÁGENES

        # --- CABEZA KAN (El cerebro) ---
        self.taylor_approx = TaylorSeriesApproximation(n_terms=5)

        # La entrada es self._to_linear (calculado automáticamente)
        self.fc1 = KANLinear(self._to_linear, 256)
        self.dropout = nn.Dropout(0.5)  # Vital para evitar overfitting
        self.fc2 = KANLinear(256, 128)
        self.dropout2 = nn.Dropout(0.4)  # Añadido otro dropout para capas intermedias
        self.fc3 = KANLinear(128, num_classes)

    def _get_conv_output(self, input_shape):
        # Función auxiliar que corre una imagen vacía por las convs
        # para saber el tamaño exacto antes de la capa lineal.
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *input_shape))
        output_feat = self._forward_features(input)
        self._to_linear = output_feat.data.view(batch_size, -1).size(1)

    def _forward_features(self, x):
        # Separamos la parte convolucional para poder reusarla en _get_conv_output
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 224 -> 112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 112 -> 56
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 56 -> 28
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 28 -> 14
        return x

    def forward(self, x):
        # 1. Extraer características (Convoluciones)
        x = self._forward_features(x)

        # 2. Aplanar
        x = torch.flatten(x, 1)

        x = torch.tanh(x)  # Estabilizador para Taylor
        # garantiza que el motor matemático de los polinomios no explote.

        # 3. Transformación Taylor (Tu componente especial)
        x = self.taylor_approx(x)

        # 4. Clasificación KAN
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

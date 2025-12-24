"""
SBTAYLOR-KAN V2: Deep Residual Taylor-KAN Model
===============================================
Architecture: Custom Deep ResNet Backbone + Taylor-KAN Head
Optimized for: Tiny ImageNet Training & Transfer Learning (e.g., Cotton Leaf Defects)

Structure:
1. Feature Extractor: Custom 10-layer Residual Network (avoids vanishing gradients)
2. Bridge: Adaptive Pooling + Batch Norm (stabilizes input for Taylor series)
3. Head: Taylor Series Approx + KAN Layers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
#                   CORE KAN COMPONENTS
#          (Estas clases se mantienen originales)
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
#            NUEVO BACKBONE: CUSTOM RESNET
# =========================================================


class CustomResBlock(nn.Module):
    """
    Bloque Residual Básico: Conv -> BN -> ReLU -> Conv -> BN -> Sum -> ReLU
    Evita el desvanecimiento del gradiente en redes profundas.
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(CustomResBlock, self).__init__()
        # Conv1: Puede reducir dimensión espacial si stride > 1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Conv2: Mantiene dimensión
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Si cambiamos dimensiones, ajustamos la identidad (skip connection)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # La clave del éxito: Suma residual
        out = self.relu(out)

        return out


# =========================================================
#                  MAIN NETWORK (MODIFIED)
# =========================================================


class DeepTaylorKAN(nn.Module):
    """
    Reemplazo de 'Net'. Arquitectura profunda híbrida.
    Feature Extractor: Custom ResNet (Diseño propio)
    Head: Taylor + KAN
    """

    def __init__(self, num_classes=10, input_channels=3, dropout=0.5):
        super(DeepTaylorKAN, self).__init__()

        self.in_channels = 64

        # --- 1. STEM (Entrada) ---
        # Procesa la imagen cruda
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # --- 2. BACKBONE RESIDUAL ---
        # Layer 1: 64 canales (Detalles finos: venas, bordes)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        # Layer 2: 128 canales (Formas simples)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        # Layer 3: 256 canales (Texturas complejas, manchas)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        # Layer 4: 512 canales (Conceptos abstractos, patologías)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        # --- 3. BRIDGE (Adaptación) ---
        # AdaptiveAvgPool reduce cualquier tamaño (H, W) a (1, 1)
        # Salida garantizada: [Batch, 512, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # BatchNorm1d antes de Taylor es CRITICO para estabilidad numérica
        self.bn_bridge = nn.BatchNorm1d(512)

        # --- 4. HEAD (Taylor + KAN) ---
        self.taylor_approx = TaylorSeriesApproximation(n_terms=5)

        # Clasificador KAN
        # Entrada siempre es 512 gracias al avgpool
        self.kan_fc1 = KANLinear(512, 128, grid_size=5)
        self.dropout = nn.Dropout(dropout)
        self.kan_fc2 = KANLinear(128, num_classes, grid_size=5)

        # Inicialización de pesos (He init para CNN)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride):
        """Construye una secuencia de Bloques Residuales"""
        downsample = None
        # Si stride != 1 o cambiamos canales, necesitamos ajustar la 'identity'
        # para que se pueda sumar al final del bloque
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(
            CustomResBlock(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(CustomResBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. Extracción de características (Deep ResNet)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # [Batch, 64, H, W]
        x = self.layer2(x)  # [Batch, 128, H/2, W/2]
        x = self.layer3(x)  # [Batch, 256, H/4, W/4]
        x = self.layer4(x)  # [Batch, 512, H/8, W/8]

        # 2. Adaptación
        x = self.avgpool(x)  # [Batch, 512, 1, 1]
        x = torch.flatten(x, 1)  # [Batch, 512]

        # 3. Estabilización y Taylor
        x = self.bn_bridge(x)  # Normalizar antes de func. no lineales complejas
        x = torch.tanh(x)  # Acotar a [-1, 1]
        x = self.taylor_approx(x)  # Inyección de priors funcionales

        # 4. Clasificación KAN
        x = self.kan_fc1(x)
        x = F.silu(x)  # SiLU es la activación nativa preferida de KAN
        x = self.dropout(x)
        x = self.kan_fc2(x)

        return x

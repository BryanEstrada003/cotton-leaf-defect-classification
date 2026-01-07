# app/gradcam_kan.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2


def generate_gradcam_kan(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    target_size=(224, 224),
):
    """
    Grad-CAM aplicado al bloque CNN del modelo KAN
    (igual que en el paper).
    """

    model.eval()

    # 🔹 Última capa convolucional
    target_layer = model.features[-2]

    activations = []
    gradients = []

    # 🔹 Hooks
    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # 🔹 Forward
    outputs = model(input_tensor)
    score = outputs[:, class_idx]

    # 🔹 Backward
    model.zero_grad()
    score.backward()

    # 🔹 Obtener activaciones y gradientes
    acts = activations[0]          # [1, C, H, W]
    grads = gradients[0]           # [1, C, H, W]

    # 🔹 Pesos α_k
    weights = grads.mean(dim=(2, 3), keepdim=True)

    # 🔹 Grad-CAM
    cam = (weights * acts).sum(dim=1)
    cam = F.relu(cam)

    # 🔹 Normalizar
    cam = cam.squeeze().detach().cpu().numpy()
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    # 🔹 Resize a tamaño original
    cam = cv2.resize(cam, target_size)

    # 🔹 Colormap
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 🔹 Limpiar hooks
    h1.remove()
    h2.remove()

    return heatmap

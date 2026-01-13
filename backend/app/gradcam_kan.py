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
    Grad-CAM adaptado específicamente para la arquitectura SBTAYLOR-KAN.
    Resalta las regiones críticas que influyen en la clasificación.
    """

    model.eval()

    # 🔹 Capa objetivo ajustada a tu arquitectura:
    # En tu modelo 'Net', model.features[12] es la última capa Conv2d.
    target_layer = model.features[12]

    activations = []
    gradients = []

    # 🔹 Hooks para capturar activaciones y gradientes [cite: 1288]
    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_input, grad_output):
        # Captura los gradientes que fluyen hacia la última capa convolucional [cite: 26]
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # 🔹 Forward pass
    outputs = model(input_tensor)
    
    # Si no se proporciona class_idx, usamos la clase con mayor probabilidad
    if class_idx is None:
        class_idx = outputs.argmax(dim=1).item()
        
    score = outputs[:, class_idx]

    # 🔹 Backward pass para calcular gradientes respecto a la clase objetivo [cite: 1288]
    model.zero_grad()
    score.backward()

    # 🔹 Obtener activaciones y gradientes
    acts = activations[0]          # [1, 128, 14, 14]
    grads = gradients[0]           # [1, 128, 14, 14]

    # 🔹 Pesos α_k: Global Average Pooling de los gradientes [cite: 1288]
    weights = grads.mean(dim=(2, 3), keepdim=True)

    # 🔹 Combinación lineal ponderada (Grad-CAM)
    cam = (weights * acts).sum(dim=1)
    
    # 🔹 ReLU para resaltar solo las características con influencia positiva [cite: 1305]
    cam = F.relu(cam)

    # 🔹 Normalizar el mapa de calor
    cam = cam.squeeze().detach().cpu().numpy()
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    # 🔹 Redimensionar al tamaño de la imagen original (target_size)
    cam = cv2.resize(cam, target_size)

    # 🔹 Generar mapa de calor visual (Heatmap) [cite: 26, 1294]
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 🔹 Limpiar hooks para evitar fugas de memoria o errores en llamadas futuras
    h1.remove()
    h2.remove()

    return heatmap
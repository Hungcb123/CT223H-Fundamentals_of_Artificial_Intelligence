import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import base64
import io

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    giúp trực quan hóa vùng đặc trưng mà mạng ResNet18 tập trung vào.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.layer4[-1]
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class_idx):
        """
        Tính toán heatmap từ gradient và activation map của layer4.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        target_score = output[0, target_class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Không thể trích xuất gradients hoặc activations từ model")

        # Global average pooling của gradients làm trọng số
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-7:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam

    def create_visualization(self, pil_image: Image.Image, heatmap: np.ndarray, alpha=0.45):
        """
        Trộn heatmap JET màu vào ảnh X-ray gốc.
        """
        img_resized = pil_image.convert('RGB').resize((224, 224))
        img_np = np.array(img_resized)

        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(img_np, 1.0 - alpha, colored_heatmap, alpha, 0)
        return Image.fromarray(blended), Image.fromarray(colored_heatmap)

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

def image_to_base64(img: Image.Image, img_format='JPEG') -> str:
    """Chuyển PIL Image thành Base64 Data URL"""
    buffer = io.BytesIO()
    img.save(buffer, format=img_format, quality=90)
    b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/{img_format.lower()};base64,{b64_str}"

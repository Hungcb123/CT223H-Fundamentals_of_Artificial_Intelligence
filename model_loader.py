import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# Tiền xử lý ảnh tương thích tuyệt đối với tập huấn luyện
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_pneumonia_model(weights_path='resnet18_pretrained_best.pth', device='cpu'):
    """
    Khởi tạo kiến trúc ResNet18 và nạp bộ trọng số đã huấn luyện tốt nhất.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy file trọng số tại: {weights_path}")

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Định dạng checkpoint không hợp lệ")

    model.to(device)
    model.eval()
    return model

def predict_pil_image(model, image: Image.Image, device='cpu'):
    """
    Thực hiện suy luận trên đối tượng PIL Image.
    Trả về dict chứa nhãn, độ tin cậy, và xác suất của từng lớp.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        normal_prob = float(probabilities[0].item())
        pneumonia_prob = float(probabilities[1].item())
        predicted_idx = int(torch.argmax(probabilities).item())
        predicted_label = CLASS_NAMES[predicted_idx]
        confidence = pneumonia_prob if predicted_idx == 1 else normal_prob

    return {
        'predicted_label': predicted_label,
        'predicted_idx': predicted_idx,
        'confidence': round(confidence * 100, 2),
        'normal_prob': round(normal_prob * 100, 2),
        'pneumonia_prob': round(pneumonia_prob * 100, 2),
        'tensor': tensor
    }

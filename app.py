import os
import time
import io
from dotenv import load_dotenv

# Nạp các biến môi trường từ file .env nếu có
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model_loader import load_pneumonia_model, predict_pil_image
from gradcam import GradCAM, image_to_base64

app = FastAPI(
    title="PneumoScan AI - Chẩn đoán Viêm phổi từ X-quang ngực",
    description="Hệ thống hỗ trợ chẩn đoán viêm phổi sử dụng ResNet18 Transfer Learning và Grad-CAM Explainable AI",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "resnet18_pretrained_best.pth")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Khởi tạo model và GradCAM
print(">>> Đang nạp mô hình ResNet18 pretrained checkpoint...")
model = load_pneumonia_model(WEIGHTS_PATH, device="cpu")
gradcam = GradCAM(model)
print(">>> Mô hình đã sẵn sàng hoạt động!")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.get("/")
def read_root():
    """Trang chủ hiển thị Web App"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "PneumoScan AI API is running"}


@app.get("/api/samples")
def get_sample_images():
    """Lấy danh sách các ảnh mẫu có sẵn"""
    return [
        {
            "id": "normal",
            "name": "Bệnh nhân A (Phổi bình thường)",
            "url": "/samples/normal_sample.jpg",
            "expected": "NORMAL",
            "description": "Phế trường hai bên sáng đều, rốn phổi bình thường, góc sườn hoành nhọn rõ."
        },
        {
            "id": "pneumonia",
            "name": "Bệnh nhân B (Viêm phổi cấp)",
            "url": "/samples/pneumonia_sample.jpg",
            "expected": "PNEUMONIA",
            "description": "Hình ảnh đông đặc mờ phế trường, thâm nhiễm phế nang điển hình của viêm phổi."
        }
    ]


@app.post("/api/predict")
async def predict_xray(
    file: UploadFile = File(None),
    sample_id: str = Form(None)
):
    """
    Tiếp nhận ảnh X-quang (qua upload hoặc qua sample_id)
    Trả về chẩn đoán, độ tin cậy và ảnh nhiệt Grad-CAM
    """
    img = None
    start_time = time.time()

    try:
        if sample_id:
            sample_filename = f"{sample_id}_sample.jpg"
            sample_path = os.path.join(SAMPLES_DIR, sample_filename)
            if not os.path.exists(sample_path):
                raise HTTPException(status_code=404, detail=f"Không tìm thấy mẫu: {sample_id}")
            img = Image.open(sample_path)
        elif file:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Tệp tải lên rỗng")
            img = Image.open(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Vui lòng tải lên 1 tệp ảnh hoặc chọn ảnh mẫu")

        # Đảm bảo định dạng RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 1. Dự đoán bằng ResNet18
        res = predict_pil_image(model, img, device="cpu")
        tensor = res["tensor"]
        pred_idx = res["predicted_idx"]

        # 2. Tạo Grad-CAM Explainable AI Heatmap
        heatmap = gradcam.generate(tensor, pred_idx)
        blended_img, pure_heatmap_img = gradcam.create_visualization(img, heatmap, alpha=0.45)

        # 3. Chuẩn bị ảnh Base64 trả về UI
        original_b64 = image_to_base64(img.resize((450, 450)))
        gradcam_b64 = image_to_base64(blended_img.resize((450, 450)))
        pure_heatmap_b64 = image_to_base64(pure_heatmap_img.resize((450, 450)))

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        # 4. Tạo khuyến cáo y khoa sơ bộ
        if res["predicted_label"] == "PNEUMONIA":
            recommendation = (
                "Phát hiện tổn thương thâm nhiễm hoặc đông đặc phế nang trên phim X-quang. "
                "Cần kết hợp thăm khám lâm sàng (nghe ran nổ, sốt, khó thở) và xét nghiệm CRP/CT ngực nếu cần thiết."
            )
            severity = "Nguy cơ cao" if res["confidence"] > 80 else "Nghi ngờ cần kiểm tra lại"
        else:
            recommendation = (
                "Không ghi nhận tổn thương dạng đông đặc hoặc thâm nhiễm khu trú trên trường phổi. "
                "Phế trường thông khí tốt, các mốc giải phẫu trung thất và vòm hoành trong giới hạn bình thường."
            )
            severity = "Bình thường"

        return JSONResponse({
            "status": "success",
            "predicted_label": res["predicted_label"],
            "confidence": res["confidence"],
            "normal_prob": res["normal_prob"],
            "pneumonia_prob": res["pneumonia_prob"],
            "severity": severity,
            "recommendation": recommendation,
            "inference_time_ms": elapsed_ms,
            "images": {
                "original": original_b64,
                "gradcam": gradcam_b64,
                "heatmap": pure_heatmap_b64
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Lỗi xử lý ảnh: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    print("Khởi chạy PneumoScan AI Server tại http://localhost:8000 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

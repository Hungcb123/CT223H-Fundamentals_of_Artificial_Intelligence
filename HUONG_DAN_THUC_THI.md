# Pneumonia Classification from Chest X-ray Images Using Transfer Learning with ResNet18

## Hướng dẫn thực thi — CT223H

---

## Tổng quan đề tài

| Mục | Chi tiết |
|---|---|
| **Tên đề tài** | Pneumonia Classification from Chest X-ray Images Using Transfer Learning with ResNet18 |
| **Bài toán** | Binary Image Classification (phân loại ảnh nhị phân) |
| **Classes** | NORMAL (0), PNEUMONIA (1) |
| **Model** | ResNet18 + ImageNet pretrained weights |
| **Dataset** | Kermany Chest X-ray Pneumonia Dataset (~5,856 images) |
| **Input** | Ảnh X-ray ngực 224 x 224 x 3 (RGB) |
| **Output** | Xác suất NORMAL hoặc PNEUMONIA |
| **Môi trường** | Google Colab (GPU T4) |

### Research Questions

1. How effectively can a pretrained ResNet18 model distinguish between normal and pneumonia chest X-ray images?
2. How does transfer learning affect pneumonia classification performance compared with training ResNet18 from scratch?

---

## Bước 1: Chuẩn bị môi trường

### Google Colab

1. Mở Google Colab: https://colab.research.google.com/
2. Runtime → Change runtime type → GPU (T4)
3. Upload file `Pneumonia_Classification_ResNet18.ipynb` lên Colab

### Kaggle API Key & Token (Chọn 1 trong 2 cách cực dễ)

Giao diện Kaggle hiện tại có 2 lựa chọn:

**Cách A — Tải file `kaggle.json` (Nhanh nhất):**
1. Vào https://www.kaggle.com/settings (tab **API Tokens**)
2. Nhìn ngay xuống phía dưới bảng, ở mục **Legacy API Credentials**
3. Bấm vào nút **`Create Legacy API Key`**
4. Trình duyệt sẽ **tự động tải file `kaggle.json` về máy tính**!
5. Khi chạy cell trên Colab, chỉ cần bấm chọn upload file `kaggle.json` này là xong.

**Cách B — Dùng file cấu hình `.env` (Bảo mật, không lộ key trong code):**
1. Mở file `.env` ở thư mục gốc dự án (đã có sẵn hoặc copy từ `.env.example`).
2. Điền thông tin:
   - `KAGGLE_USERNAME=tên_username_kaggle`
   - `KAGGLE_KEY=chuỗi_token_hoặc_key`
3. Notebook sẽ tự động nạp `KAGGLE_USERNAME` và `KAGGLE_KEY` từ file `.env` (hoặc biến môi trường hệ thống) và cấu hình xác thực an toàn, không lo lộ key khi chia sẻ notebook.

---

## Bước 2: Download Dataset

Dataset sẽ được download tự động qua Kaggle API trong notebook.

Nếu Kaggle API không hoạt động, dùng cách backup:
1. Tải thủ công từ Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Upload file zip lên Google Drive
3. Uncomment block "CÁCH 2" trong notebook

---

## Bước 3-9: Chạy Notebook

Notebook được thiết kế chạy tuần tự từ trên xuống dưới.
Chỉ cần Run All hoặc chạy từng cell.

### Pipeline hoàn chỉnh trong notebook:

```
Download Dataset
      |
  EDA (phân tích dữ liệu)
      |
  Re-split Train/Val (85/15)
      |
  Preprocessing + Augmentation
      |
  Build ResNet18 (pretrained)
      |
  Train + Early Stopping
      |
  Evaluate on Test Set
      |
  Experiment 2: From Scratch
      |
  So sánh 2 Experiments
      |
  Error Analysis (FP/FN)
      |
  Final Summary + Export
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Image size | 224 x 224 |
| Batch size | 32 |
| Epochs | 15 (max, with early stopping) |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Loss | Weighted CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |
| Early stopping | patience=4 |

---

## Output Files

### Models
- resnet18_pretrained_best.pth
- resnet18_scratch_best.pth

### Hình ảnh cho báo cáo
1. class_distribution.png — Phân phối class
2. sample_images.png — Ảnh X-ray mẫu
3. training_history.png — Training curves
4. confusion_matrix_pretrained.png — Confusion matrix
5. roc_curve_pretrained.png — ROC curve
6. sample_predictions.png — Ảnh dự đoán mẫu
7. comparison_results.png — So sánh 2 experiments
8. confusion_matrices_comparison.png — So sánh confusion matrices
9. roc_curves_comparison.png — So sánh ROC curves
10. false_negatives.png — Các ca bỏ sót pneumonia
11. false_positives.png — Các ca dương tính giả

### Data
- experiment_results.csv — Bảng so sánh metrics

---

## Evaluation Metrics

| Metric | Ý nghĩa |
|---|---|
| Accuracy | Tỷ lệ dự đoán đúng tổng thể |
| Precision | Trong các ca dự đoán Pneumonia, bao nhiêu % đúng? |
| Recall/Sensitivity | Trong các ca thật sự Pneumonia, model phát hiện được bao nhiêu %? |
| Specificity | Trong các ca Normal thật sự, model nhận đúng bao nhiêu %? |
| F1-Score | Trung bình điều hòa Precision và Recall |
| ROC-AUC | Diện tích dưới đường cong ROC |

---

## Cấu trúc báo cáo gợi ý

1. Introduction — Bài toán, tại sao cần AI trong chẩn đoán
2. Related Work — Transfer learning, ResNet, medical imaging
3. Dataset — Kermany dataset, class distribution, re-split
4. Methodology — ResNet18, transfer learning, preprocessing
5. Experiments — Pretrained vs From Scratch
6. Results — Metrics, confusion matrix, ROC curve
7. Error Analysis — Phân tích FP/FN
8. Conclusion — Kết luận, hạn chế, hướng phát triển

---

## Timeline ước tính

| Bước | Thời gian |
|---|---|
| Setup + Download | 5-10 phút |
| EDA + Preprocessing | 2 phút |
| Training Exp.1 (pretrained) | 10-15 phút |
| Training Exp.2 (scratch) | 10-15 phút |
| Evaluation + So sánh | 3 phút |
| **Tổng** | **~30-45 phút** |

---

## Troubleshooting

- Kaggle API lỗi → Dùng cách 2: upload qua Google Drive
- Colab hết GPU → Terminate other sessions, đợi ít phút
- Out of memory → Giảm BATCH_SIZE từ 32 xuống 16
- Training chậm → Kiểm tra Runtime type có phải GPU không

---

*CT223H — Nền tảng trí tuệ nhân tạo*

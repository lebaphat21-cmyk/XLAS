# Phát hiện Ảnh và Video Giả mạo do AI (Deepfake and AI-Generated Detection)

Đồ án xây dựng hệ thống phát hiện ảnh và video giả mạo kết hợp đặc trưng **Không gian (Spatial)**, **Tần số (Frequency)**, và **Thời gian (Temporal)**.

## Cấu trúc thư mục
- `configs/config.yaml`: Cấu hình siêu tham số và đường dẫn.
- `src/preprocessing.py`: Phát hiện khuôn mặt, crop, biến đổi tần số FFT/DCT.
- `src/dataset.py`: Quản lý dữ liệu PyTorch Dataset & DataLoader.
- `src/models.py`: Khai báo các mô hình Baseline (SVM, CNN RGB) và Mô hình đề xuất (Fusion, BiGRU + Attention).
- `src/train.py`: Huấn luyện mô hình ảnh và video.
- `src/eval.py`: Đánh giá hiệu suất và độ bền (nén JPEG, nén video).
- `src/explain.py`: Giải thích mô hình bằng Grad-CAM và biểu đồ tần số.
- `app.py`: Web Demo tương tác Streamlit.

## Cách chạy dự án

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu và Tiền xử lý
Tổ chức dữ liệu trong thư mục `data/raw/` theo cấu trúc:
```
data/raw/
├── real/
│   ├── image1.jpg
│   └── video1.mp4
└── fake/
    ├── image2.jpg
    └── video2.mp4
```
Chạy tiền xử lý dữ liệu:
```bash
python -m src.preprocessing
```

### 3. Huấn luyện mô hình
Huấn luyện mô hình ảnh (RGB-Frequency Fusion):
```bash
python -m src.train --mode image
```
Huấn luyện mô hình video (BiGRU + Temporal Attention):
```bash
python -m src.train --mode video
```

### 4. Đánh giá mô hình
```bash
python -m src.eval
```

### 5. Chạy Demo
```bash
streamlit run app.py
```

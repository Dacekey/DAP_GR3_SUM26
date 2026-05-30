# 👁️ Eye Blink Detection System – Health Monitor

> Hệ thống nhận diện số lần nháy mắt khi sử dụng laptop/máy tính để cảnh báo sức khỏe người dùng.  
> **Môn học**: DAP391m – Computer Vision (Machine Learning only, no Deep Learning)  
> **Nhóm**: 6 thành viên

---

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Pipeline](#pipeline)
- [Cài Đặt](#cài-đặt)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Thuật Toán & Phương Pháp](#thuật-toán--phương-pháp)
- [Kết Quả](#kết-quả)

---

## Tổng Quan

Hệ thống sử dụng camera laptop để theo dõi tần suất nháy mắt của người dùng trong thời gian thực, từ đó đưa ra cảnh báo sức khỏe:

| Chỉ số | Bình thường 🟢 | Cảnh báo 🟡 | Nguy hiểm 🔴 |
|--------|----------------|-------------|--------------|
| Tần suất nháy | ≥ 15 lần/phút | 10–14 lần/phút | < 10 lần/phút |
| Thời gian nhắm mắt | < 0.4 giây | 0.4–2 giây | > 2 giây (ngủ gật) |
| Thời gian sử dụng | < 30 phút | 30–60 phút | > 60 phút liên tục |

---

## Pipeline

```
Camera (ngầm) → Face Detection (dlib HOG+SVM) → Facial Landmarks (68-point)
→ Eye Extraction → Feature Extraction (EAR + HOG + LBP) → SVM Classifier
→ Blink Counting (State Machine) → Health Monitor → Mini Dashboard (tkinter)
```

**Tất cả các phương pháp đều là Machine Learning truyền thống:**
- Face Detection: HOG features + Linear SVM
- Landmark Detection: Ensemble of Regression Trees
- Classifier: SVM (RBF kernel) / Random Forest

---

## Cài Đặt

### 1. Yêu cầu hệ thống
- **Python 3.10** (bắt buộc – dlib chỉ tương thích tốt với Python 3.10)
- Webcam (camera laptop)
- Windows / macOS / Linux

### 2. Tạo virtual environment

```bash
# Tạo venv bằng Python 3.10
python3.10 -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

> **Lưu ý**: Nếu gặp lỗi khi cài `dlib`, cần cài CMake trước:
> ```bash
> pip install cmake
> pip install dlib
> ```

### 3. Tải model landmark

Tải file `shape_predictor_68_face_landmarks.dat`:

```bash
# Tải từ dlib
# Link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Giải nén và đặt vào thư mục models/
```

---

## Hướng Dẫn Sử Dụng

### Bước 1: Thu thập dataset

```bash
# Quay video từ webcam (theo quy chuẩn nhóm: 12s, 15fps, .mp4)
python data/collect_video.py

# Trích xuất ảnh mắt từ video (có alignment xoay ngang)
python data/extract_eyes.py              # Chạy bình thường
python data/extract_eyes.py --preview    # Xem preview landmarks + eye patches

# Gán nhãn bán tự động
python data/label_tool.py --mode auto    # Gán nhãn tự động bằng EAR
python data/label_tool.py --mode review  # Review và sửa nhãn thủ công
python data/label_tool.py --mode split   # Chia train/test 80/20
```

### Bước 2: Train model

```bash
python src/train_model.py               # Train cả SVM và Random Forest
python src/train_model.py --model svm   # Chỉ train SVM
python src/train_model.py --model rf    # Chỉ train Random Forest
```

### Bước 3: Chạy hệ thống

```bash
python main.py                   # Chạy với mini dashboard
python main.py --no-dashboard    # Chạy console only
python main.py --debug           # Chạy với debug video window
```

---

## Cấu Trúc Dự Án

```
eye-blink-detection/
├── data/
│   ├── collect_video.py          # Script quay video (12s, 15fps, .mp4)
│   ├── extract_eyes.py           # Trích xuất + alignment mắt
│   ├── label_tool.py             # Gán nhãn
│   └── raw_videos/               # Video thô (tự tạo)
├── dataset/
│   ├── raw_eyes/                 # Ảnh mắt trích xuất + ear_values.csv
│   ├── train/open/               # Ảnh mắt mở (train 80%)
│   ├── train/closed/             # Ảnh mắt nhắm (train 80%)
│   ├── test/open/                # Ảnh mắt mở (test 20%)
│   └── test/closed/              # Ảnh mắt nhắm (test 20%)
├── src/
│   ├── features.py               # EAR + HOG + LBP extraction
│   ├── train_model.py            # Train & evaluate models
│   ├── blink_detector.py         # Real-time blink detection
│   ├── health_monitor.py         # Health monitoring logic
│   └── dashboard.py              # tkinter mini dashboard
├── models/
│   ├── shape_predictor_68_face_landmarks.dat  # dlib model (tải riêng)
│   ├── svm_blink_model.pkl       # Trained SVM (tự tạo)
│   └── scaler.pkl                # Feature scaler (tự tạo)
├── config.py                     # Cấu hình tập trung
├── main.py                       # Entry point
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # Tài liệu này
```

---

## Thuật Toán & Phương Pháp

### Eye Aspect Ratio (EAR)

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)

    p2  p3
p1          p4
    p6  p5
```

- Mắt mở: EAR ≈ 0.25 – 0.35
- Mắt nhắm: EAR < 0.20

### Feature Vector

| Feature | Dimensions | Method |
|---------|-----------|--------|
| EAR | 3 | Eye Aspect Ratio (left, right, avg) |
| HOG | ~100-200 | Histogram of Oriented Gradients |
| LBP | ~10 | Local Binary Pattern histogram |
| **Total** | **~115-215** | Concatenated vector |

### Blink State Machine

```
OPEN → (EAR↓ + SVM=closed) → MAYBE_CLOSED
MAYBE_CLOSED → (≥2 frames closed) → CLOSED
CLOSED → (EAR↑ + SVM=open) → MAYBE_OPEN
MAYBE_OPEN → (≥2 frames open) → OPEN (blink counted!)
```

### Classifier

- **Primary**: SVM (RBF kernel) with GridSearchCV tuning
- **Comparison**: Random Forest (200 trees)
- **Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## Kết Quả

> *Phần này sẽ được cập nhật sau khi train model.*

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| SVM (RBF) | – | – | – | – |
| Random Forest | – | – | – | – |

---

## Thành Viên Nhóm

| STT | Họ Tên | Vai Trò |
|-----|--------|---------|
| 1 | | Data Collection |
| 2 | | Data Collection |
| 3 | | ML Pipeline |
| 4 | | ML Pipeline |
| 5 | | Real-time System |
| 6 | | Dashboard & Report |

---

## Tài Liệu Tham Khảo

1. Soukupová, T., & Čech, J. (2016). "Real-Time Eye Blink Detection using Facial Landmarks." *21st Computer Vision Winter Workshop*.
2. Kazemi, V., & Sullivan, J. (2014). "One Millisecond Face Alignment with an Ensemble of Regression Trees." *CVPR 2014*.
3. Dalal, N., & Triggs, B. (2005). "Histograms of Oriented Gradients for Human Detection." *CVPR 2005*.

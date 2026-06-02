# 👁️ Eye Blink Data Collection Pipeline

> Hệ thống thu thập, trích xuất và gán nhãn dữ liệu mắt (Mở/Nhắm) phục vụ cho bài toán nhận diện nháy mắt (Blink Detection).
> **Dự án**: Cảnh báo sức khỏe người dùng qua tần suất nháy mắt.
> **Trạng thái hiện tại**: Giai đoạn thu thập và tiền xử lý dữ liệu (Data Collection & Labeling).

---

## 📋 Mục Lục

- [Tổng Quan Dự Án](#tổng-quan-dự-án)
- [Kiến Trúc & Tính Năng Chức Năng](#kiến-trúc--tính-năng)
- [Cài Đặt Hệ Thống](#cài-đặt-hệ-thống)
- [Hướng Dẫn Quy Trình Thực Hiện](#hướng-dẫn-quy-trình-thực-hiện)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)

---

## Tổng Quan Dự Án

Đây là nền tảng khởi đầu cho dự án **Cảnh báo sức khỏe người dùng**. Chức năng chính của phần này là thu thập và chuẩn bị một bộ dữ liệu (dataset) chất lượng cao về hình ảnh mắt của người dùng. 

Hệ thống sử dụng **MediaPipe Face Mesh** để quét khuôn mặt qua webcam, tự động định vị và cắt chính xác khu vực mắt trái/phải thành các hình ảnh kích thước 24x24 pixel. Những hình ảnh này, kết hợp với nhãn (Nhắm/Mở) được gán thủ công hoặc tự động qua các công cụ tích hợp, sẽ đóng vai trò cốt lõi trong việc huấn luyện các mô hình Machine Learning sau này (như SVM, Random Forest).

---

## Kiến Trúc & Tính Năng

**1. Ghi hình Webcam (Video Collection):** Thu thập các luồng video thực tế của người dùng với nhiều điều kiện ánh sáng và góc độ khác nhau.
**2. Trích xuất tự động (Eye Extraction):** Bỏ qua các phương pháp truyền thống chậm chạp, hệ thống sử dụng sức mạnh của MediaPipe để tracking khuôn mặt với độ trễ cực thấp.
**3. Công cụ Gán nhãn (Labeling Tool):** Bộ công cụ console tiện dụng cho phép người dùng gán nhãn hàng loạt tự động bằng công thức EAR, duyệt lại (review) bằng phím tắt một cách trực quan, và tự động chia tách dữ liệu vào các thư mục Train/Test.

---

## Cài Đặt Hệ Thống

### 1. Yêu cầu môi trường
- **Python 3.8+** (Khuyên dùng Python 3.10)
- Webcam (camera laptop hoặc webcam rời)
- Hệ điều hành: Windows / macOS / Linux

### 2. Thiết lập dự án

```bash
# Khởi tạo virtual environment
python -m venv venv

# Kích hoạt venv (trên Windows)
venv\Scripts\activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```
> **Lưu ý**: Nhờ sử dụng thư viện hiện đại MediaPipe, việc cài đặt diễn ra cực kỳ nhanh chóng mà không cần các thao tác build phức tạp từ CMake như trước.

---

## Hướng Dẫn Quy Trình Thực Hiện

Để giữ cho tài liệu README tổng quan và dễ đọc, toàn bộ **hướng dẫn chi tiết từng bước** về cách thu thập video, cách trích xuất, và các phím tắt dùng để gán nhãn dữ liệu đã được tách riêng vào thư mục `processing`.

👉 **Vui lòng xem chi tiết tại:** [Hướng Dẫn Quy Trình Xử Lý Dữ Liệu](src/data_collection/workflow_guide.md)

*(Trong file này sẽ chứa toàn bộ các dòng lệnh cần thiết để chạy pipeline của bạn một cách trơn tru)*

---

## Cấu Trúc Thư Mục

```text
eye-blink-detection/
├── data/
│   └── raw_videos/               # Chứa video gốc (quay từ người dùng)
├── dataset/
│   ├── raw_eyes/                 # Ảnh mắt thô & metadata được tạo ra tự động
│   ├── train/                    # Bộ dữ liệu 80% để huấn luyện
│   └── test/                     # Bộ dữ liệu 20% để kiểm thử
├── src/
│   └── data_collection/          # Nơi chứa mã nguồn chính của pipeline
│       ├── collect_video.py      
│       ├── extract_eyes.py       
│       ├── label_tool.py         
│       └── workflow_guide.md     # 📖 TÀI LIỆU HƯỚNG DẪN QUY TRÌNH CHI TIẾT
├── config.py                     # File cấu hình (kích thước, EAR threshold,...)
├── requirements.txt              # Danh sách thư viện cần thiết
└── README.md                     # Tài liệu tổng quan dự án (File này)
```

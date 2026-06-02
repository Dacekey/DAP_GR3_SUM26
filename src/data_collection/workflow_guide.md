# Hướng Dẫn Quy Trình Xử Lý Dữ Liệu (Data Processing Workflow)

Quy trình xử lý dữ liệu của dự án bao gồm 3 bước chính: Thu thập video, Trích xuất ảnh mắt, và Gán nhãn dữ liệu. Bạn hãy thực hiện lần lượt các bước dưới đây.

## Bước 1: Thu thập Video (Video Collection)

Sử dụng camera để quay các đoạn video ngắn (khoảng 12 giây) người dùng đang chớp mắt hoặc nhắm mắt.
- **Lệnh thực thi**:
  ```bash
  python src/data_collection/collect_video.py
  ```
- **Kết quả**: Video sẽ được lưu dưới định dạng `.mp4` vào thư mục `data/raw_videos/`.

## Bước 2: Trích xuất hình ảnh mắt (Eye Extraction)

Hệ thống sẽ đọc các video thô, sử dụng **MediaPipe Face Mesh** để nhận diện khuôn mặt và cắt ra các vùng mắt (trái/phải) với kích thước 24x24 pixel.
- **Lệnh thực thi (Chạy ngầm)**:
  ```bash
  python src/data_collection/extract_eyes.py
  ```
- **Lệnh thực thi (Có giao diện xem trước)**:
  ```bash
  python src/data_collection/extract_eyes.py --preview
  ```
- **Kết quả**: 
  - Các hình ảnh mắt 24x24 được lưu vào thư mục `dataset/raw_eyes/`.
  - Một file `metadata.csv` được tạo ra chứa các chỉ số EAR (Eye Aspect Ratio) của từng khung hình.

## Bước 3: Gán nhãn dữ liệu (Data Labeling)

Dữ liệu hình ảnh cần được gán nhãn `Open` (Mở mắt) hoặc `Closed` (Nhắm mắt) để phục vụ cho việc huấn luyện mô hình sau này. Có 3 chế độ gán nhãn:

**1. Gán nhãn tự động (Auto Labeling):**
Dựa vào chỉ số EAR và một ngưỡng (threshold) để tự động phân loại.
```bash
python src/data_collection/label_tool.py --mode auto 
```

**2. Đánh giá và chỉnh sửa thủ công (Manual Review):**
Sau khi gán tự động, bạn nên kiểm tra lại bằng mắt thường. Giao diện review sẽ hiện ra từng ảnh mắt.
```bash
python src/data_collection/label_tool.py --mode review
```
*Các phím tắt hỗ trợ trong quá trình review:*
- `o`: Đổi nhãn thành **Open** (Mở mắt).
- `c`: Đổi nhãn thành **Closed** (Nhắm mắt).
- `d` hoặc `x`: **Xóa bỏ (Discard)** ảnh lỗi (ảnh mờ, cắt sai vị trí).
- `SPACE`: Bỏ qua (Giữ nguyên nhãn hiện tại).
- `q`: Lưu tiến độ và thoát.

**3. Chia tập Train/Test (Split Dataset):**
Sau khi hoàn tất gán nhãn, chia bộ dữ liệu vào các thư mục `train` (80%) và `test` (20%).
```bash
python src/data_collection/label_tool.py --mode split
```
- **Kết quả**: Dữ liệu sẽ được tự động phân bổ vào `dataset/train/` và `dataset/test/` sẵn sàng cho các bước Machine Learning.

---

## Bước 4: Làm việc nhóm và Gộp dữ liệu (Dành cho Leader/ML Team)

Khi làm việc nhóm, mỗi người tự thực hiện từ Bước 1 đến Bước 3 trên máy cá nhân. Sau đó, nhóm cần gom dữ liệu lại để có một Master Dataset.

**4.1. Nộp dữ liệu lên Google Drive (Upload):**
1. Tạo một thư mục chung trên Google Drive (VD: `EyeBlink_Contributions`).
2. Bên trong thư mục chung, tạo các thư mục con mang mã số hoặc tên của từng thành viên (VD: `M01/`, `M02/`...).
3. Mỗi thành viên chạy quy trình (quay video, trích xuất và gán nhãn) trên máy cá nhân của mình. Sau khi hoàn thành xong **Bước 3 (Split)**, mỗi người tự tải lên (hoặc nén zip rồi tải lên) hai thư mục dữ liệu cục bộ vào thư mục của mình trên Drive:
   - Thư mục `data/raw_videos/` (Video gốc của bạn)
   - Thư mục `dataset/` (Ảnh mắt đã cắt và nhãn hoàn chỉnh của bạn)

**Cấu trúc thư mục trên Google Drive chung của nhóm:**
```text
Google Drive/
├── M01/ (Thành viên 1)
│   ├── data/
│   │   └── raw_videos/           # Video thô của M01
│   └── dataset/                  # Ảnh mắt đã cắt và file metadata.csv của M01
├── M02/ (Thành viên 2)
│   ├── data/
│   │   └── raw_videos/
│   └── dataset/
└── ... (M03, M04, M05, M06)
```

**4.2. Gộp dữ liệu tự động (Merge):**
Người phụ trách tổng hợp dữ liệu (Leader/ML Team) sẽ tải toàn bộ các thư mục đóng góp của các thành viên từ Google Drive về máy cá nhân của mình, giải nén và đặt vào thư mục `data/contributions/` trong dự án theo cấu trúc:
```text
data/contributions/
├── M01/
│   └── dataset/  <-- Thư mục dataset của M01 (phải chứa metadata.csv)
├── M02/
│   └── dataset/  <-- Thư mục dataset của M02 (phải chứa metadata.csv)
└── ...
```
Sau đó, chạy công cụ gộp dữ liệu tại thư mục gốc của dự án:
```bash
python src/data_collection/merge_datasets.py
```
- **Kết quả**: 
  - Tạo ra một thư mục `dataset_master/` chứa toàn bộ ảnh của cả nhóm đã được gom chung và phân bổ vào các thư mục con `train` và `test` phù hợp.
  - Tạo ra một file `metadata_master.csv` tổng hợp tất cả dòng dữ liệu của cả nhóm, đồng thời tự động thêm cột `contributor` để theo dõi nguồn gốc dữ liệu của từng thành viên.
  - Thư mục `dataset_master/` này sẽ được dùng để thực hiện huấn luyện các mô hình Machine Learning sau này mà không lo bị trùng lặp hay ghi đè file!

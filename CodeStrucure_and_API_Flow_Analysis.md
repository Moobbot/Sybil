# Cấu Trúc Code và Luồng API

## 1. Cấu Trúc Thư Mục & File Chính

```text
/dicom/Sybil/
│
├── api.py                # Điểm khởi động Flask API, đăng ký blueprint
├── call_model.py         # Hàm chính để tải model, dự đoán, xử lý attention
├── config.py             # Cấu hình chung (đường dẫn, tham số model, v.v.)
├── utils.py              # Tiện ích: upload, xử lý file, dọn dẹp, zip, v.v.
│
└── sybil/
    ├── model.py          # Định nghĩa class Sybil (model chính)
    ├── serie.py          # Định nghĩa class Serie (xử lý chuỗi ảnh)
    ├── parsing.py        # Xử lý parsing DICOM, v.v.
    ├── utils/            # Tiện ích phụ trợ (logging, visualization, ...)
    ├── datasets/         # Xử lý dataset, tiền xử lý, ...
    ├── loaders/          # Hàm load dữ liệu, augmentations, ...
    └── ...               # Các module khác
```

### a. API Layer (`api.py`)

- Tạo Flask app
- Đăng ký blueprint từ `routes.py`
- Dọn dẹp các thư mục upload/result cũ khi khởi động
- Chạy Flask server trên host/port cấu hình

```py
app = Flask(__name__)
app.register_blueprint(bp)
cleanup_old_results([UPLOAD_FOLDER, RESULTS_FOLDER])
app.run(host=HOST_CONNECT, port=PORT_CONNECT, debug=True)
```

### b. Upload file & Xử lý file (`utils.py`)

- **Upload file**: Lưu file upload vào thư mục theo session_id
- **Giải nén ZIP**: Nếu upload là file zip, giải nén và lấy danh sách file hợp lệ (DICOM/PNG)
- **Dọn dẹp**: Xóa các thư mục cũ sau một thời gian
- **Chuyển đổi DICOM → PNG**: Hỗ trợ preview ảnh
- **Tạo file zip kết quả**: Nén kết quả trả về cho client

### c. Dự đoán & Xử Lý Model (`call_model.py`)

**Luồng chính khi gọi dự đoán:**

- 1.Tải model (nếu chưa có):
  - Kiểm tra checkpoint, nếu thiếu thì tải về từ GitHub
  - Tạo object Sybil từ checkpoint
- 2.Lấy danh sách file ảnh đầu vào:
  - Lấy tất cả file hợp lệ trong thư mục upload
  - Xác định loại file (DICOM/PNG) dựa vào extension
- Tạo object Serie:
  - Đóng gói chuỗi ảnh thành một Serie để đưa vào model
- Gọi model.predict:
  - Trả về prediction (score, attention, ...)
- Lưu kết quả:
  - Lưu prediction ra file JSON
  - Nếu có attention, lưu attention ra file pickle, ranking ra file JSON
- Xử lý attention (nếu có):
  - Xếp hạng ảnh theo attention score
  - Tạo thông tin trả về về các ảnh có attention cao nhất
- Visualize attention (nếu có):
  - Tạo ảnh overlay attention, lưu ra thư mục output
- Trả về kết quả:
  - Trả về dict prediction, overlay attention, info attention

### Cấu Hình (`config.py`)

- Định nghĩa các tham số cấu hình:
  - Port, host
  - Đường dẫn upload/result
  - Checkpoint
  - Các tham số model
- Đường dẫn file kết quả, overlay, attention, ranking

## 2. Luồng API Chi Tiết

### Endpoint `/predict`

1. Client upload file (DICOM/ZIP/PNG) lên endpoint
2. API nhận file, lưu vào thư mục upload theo session_id
3. Nếu là ZIP, giải nén và lấy danh sách file hợp lệ
4. Gọi hàm dự đoán (call_model.predict):
   - Tải model nếu cần
   - Tạo Serie
   - Dự đoán
   - Lưu kết quả
   - Xử lý attention, visualize nếu cần
5. Tạo file zip kết quả
6. Trả về kết quả cho client (link zip/kết quả JSON)

## 3. Các Class & Module Chính

### Sybil (`sybil/model.py`)

- Model chính
- Hàm predict
- Xử lý attention

### Serie (`sybil/serie.py`)

- Đóng gói chuỗi ảnh
- Xử lý tiền xử lý
- Validation

### Utility Modules

- `utils_datasets`: Xử lý dataset
- `visualization`: Visualize attention
- `logging_utils`: Logging

## 4. Sơ Đồ Luồng Xử Lý

```mermaid
flowchart TD
    A[Client upload file] --> B[API nhận file]
    B --> C[Lưu vào uploads/session_id]
    C --> D{ZIP?}
    D -- Yes --> E[Giải nén, lấy file hợp lệ]
    D -- No --> F[Lấy file hợp lệ]
    E --> G[Gọi call_model.predict]
    F --> G
    G --> H[Lưu kết quả, attention, visualize]
    H --> I[Tạo file zip kết quả]
    I --> J[Trả về client (link zip/kết quả JSON)]
```

## 5. Các Điểm Mở Rộng/Tùy Chỉnh

### Thêm Endpoint

- Preview ảnh
- Lấy attention
- Custom predictions

### Cấu Hình

- Số lượng ảnh top attention trả về
- Loại file được chấp nhận
- Tham số model

### Mở Rộng Model

- Hỗ trợ nhiều loại model/checkpoint
- Tùy chỉnh architecture
- Tích hợp model mới

## 6. Lưu Ý Quan Trọng

### Quản Lý File

- Lưu trữ theo session
- Tự động dọn dẹp
- Xử lý file an toàn

### Tích Hợp Model

- Lazy loading
- Kiểm tra checkpoint
- Xử lý lỗi

### Xử Lý Response

- Định dạng chuẩn
- Mã lỗi
- Tài liệu API

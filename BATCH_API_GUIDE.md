# Hướng dẫn sử dụng Batch Processing API cho Sybil

## 1. Endpoint

**POST** `/api_batch_process`

API xử lý batch: đọc tuần tự từng subfolder trong folder được chọn và thực hiện dự đoán lung cancer risk.

## 2. Request Body

```json
{
  "folder_path": "C:/path/to/folder/containing/subfolders"
}
```

## 3. Response

Trả về file ZIP chứa:
- Mỗi subfolder có 1 file CSV riêng với kết quả từng ảnh
- 1 file summary CSV tổng hợp

## 4. Cấu trúc CSV cho mỗi subfolder

Mỗi CSV chứa thông tin từng ảnh:
- `file_name`: Tên file ảnh
- `attention_score`: Điểm attention của ảnh
- `overall_score`: Điểm lung cancer risk tổng thể của subfolder
- `subfolder_name`: Tên subfolder
- `subfolder_path`: Đường dẫn đầy đủ

## 5. Cấu trúc Summary CSV

- `subfolder_name`: Tên subfolder
- `status`: "success" hoặc "error"
- `overall_score`: Điểm dự đoán lung cancer risk
- `total_images`: Tổng số ảnh đã xử lý
- `returned_images`: Số ảnh được trả về
- `error_message`: Thông báo lỗi (nếu có)
- `csv_file`: Tên file CSV tương ứng

## 6. Cách sử dụng

### Sử dụng cURL

```bash
curl -X POST "http://localhost:5555/api_batch_process" \
     -H "Content-Type: application/json" \
     -d "{\"folder_path\": \"C:/path/to/your/folder\"}" \
     --output batch_results.zip
```

### Sử dụng Python requests

```python
import requests

url = "http://localhost:5555/api_batch_process"
payload = {
    "folder_path": "C:/path/to/your/folder"
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    # Lưu file ZIP
    with open("batch_results.zip", "wb") as f:
        f.write(response.content)
    print("✅ Thành công! File ZIP đã được lưu.")
else:
    print(f"❌ Error: {response.json()}")
```

### Sử dụng JavaScript (fetch)

```javascript
fetch('http://localhost:5555/api_batch_process', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    folder_path: 'C:/path/to/your/folder'
  })
})
.then(response => response.blob())
.then(blob => {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'batch_results.zip';
  a.click();
});
```

## 7. Cấu trúc Folder

Folder bạn chọn nên có cấu trúc như sau:

```
your_folder/
├── subfolder1/
│   ├── image1.dcm
│   ├── image2.dcm
│   └── ...
├── subfolder2/
│   ├── image1.dcm
│   ├── image2.dcm
│   └── ...
└── subfolder3/
    ├── image1.dcm
    └── ...
```

## 8. Ví dụ CSV Output

### CSV cho subfolder (subfolder1_results.csv):

```csv
file_name,attention_score,overall_score,subfolder_name,subfolder_path
pred_patient1_000.dcm,0.85,0.72,subfolder1,C:/data/subfolder1
pred_patient1_001.dcm,0.78,0.72,subfolder1,C:/data/subfolder1
pred_patient1_002.dcm,0.65,0.72,subfolder1,C:/data/subfolder1
```

### Summary CSV (batch_20241201_120000_summary.csv):

```csv
subfolder_name,status,overall_score,total_images,returned_images,error_message,csv_file
subfolder1,success,0.72,120,45,,subfolder1_results.csv
subfolder2,success,0.68,98,38,,subfolder2_results.csv
subfolder3,error,,0,0,No valid DICOM/PNG files found,subfolder3_results.csv
```

## 9. Lưu ý

- **Timeout**: Quá trình xử lý có thể mất nhiều thời gian. Đảm bảo client có timeout đủ lớn (ví dụ: 1 giờ).
- **Đường dẫn**: 
  - Windows: Sử dụng `C:/path/to/folder` hoặc `C:\\path\\to\\folder`
  - Linux/macOS: Sử dụng `/path/to/folder`
- **Quyền truy cập**: Đảm bảo API có quyền đọc folder và các file bên trong.
- **Model**: Đảm bảo model đã được load thành công khi khởi động API.

## 10. Troubleshooting

### Lỗi: "Folder not found"
- Kiểm tra đường dẫn có đúng không
- Kiểm tra quyền truy cập folder

### Lỗi: "No subfolders found"
- Đảm bảo folder chứa ít nhất một subfolder
- Kiểm tra subfolder là thư mục, không phải file

### Lỗi: "Model not loaded"
- Đảm bảo model đã được load khi khởi động API
- Kiểm tra checkpoint files có tồn tại không

### Lỗi: Connection timeout
- Tăng timeout trong client
- Kiểm tra API vẫn đang chạy
- Xử lý batch có thể mất nhiều thời gian


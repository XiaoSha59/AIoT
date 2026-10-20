# Model cho demo EdgeFace

Các file ONNX được quản lý bằng **Git LFS**:

- `face_detection_yunet_2023mar.onnx`: detector pretrained; nguồn/checksum trong `.source.json`, giấy phép upstream trong [LICENSE.yunet](LICENSE.yunet).
- `releases/5010dc47644c/`: classifier hiện tại gồm `Unknown`, `dung`, `hung`, `son`, `thanh`, `tuan`; ONNX `1e170d7902b8`, bổ sung ảnh camera Sơn, kiểm tra trên Pi với 181 crop test thật. [Chi tiết sửa màu và training](../report/SON_CAMERA_UPDATE_20261020.md).
- `releases/024f31fdc9b8/`: bản team ban đầu dùng video, ONNX `409a4b48a74f`, giữ để đối chiếu/rollback.
- `releases/d699b7c900ea/`: bản demo `son` / `Unknown` trước đây, giữ để rollback.
- `active` là symlink tương đối tới release hiện tại; clone trên WSL/Linux để giữ symlink.

Sau khi clone: `git lfs install` rồi `git lfs pull`. Không dùng file pointer LFS làm input ONNX Runtime. Mỗi release kèm `classes.json`, `preprocessing.json`, `threshold.json`, `export_validation.json`. Kết quả và giới hạn đánh giá cùng phiên quay nằm trong [báo cáo training 20/10/2026](../report/TEAM_TRAINING_20261020.md).

`pretrained/`, `video2/` và release ngoài allowlist là cache/bản sao cục bộ, không đưa lên Git. Checkpoint training ở `runs/` và dữ liệu mặt/video ở `dataset/` cũng được loại khỏi Git.

Khi bổ sung dữ liệu, train lại, calibration validation, export vào thư mục mới, kiểm tra và deploy bằng `scripts/deploy_model.py`. `training.verify_onnx` kiểm tra cùng dữ liệu test trên laptop/Pi, SHA256, logits và quyết định rejection. Chỉ chia sẻ release đã xác minh; cập nhật allowlist release trong `.gitignore` và symlink `active` cùng một commit. Các lớp, preprocessing và threshold phải đi cùng ONNX, tránh trộn bundle.

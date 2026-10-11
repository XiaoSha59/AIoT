# Model cho demo EdgeFace

Hai model ONNX được lưu bằng **Git LFS**:

- `face_detection_yunet_2023mar.onnx`: detector pretrained; nguồn/checksum trong `.source.json`, giấy phép upstream trong [LICENSE.yunet](LICENSE.yunet).
- `releases/d699b7c900ea/`: classifier demo `son` / `Unknown`, kèm `classes.json`, `preprocessing.json`, `threshold.json` và `export_validation.json`.
- `active` là symlink tương đối tới release hiện tại; clone trên WSL/Linux để giữ symlink.

Sau khi clone: `git lfs install` rồi `git lfs pull`. Không dùng file pointer LFS làm input ONNX Runtime. Model này dành cho demo một thành viên, không đại diện kết quả đánh giá cuối của cả team.

`pretrained/`, `video2/` và các release cũ chỉ là cache/bản sao/rollback cục bộ, không đưa lên Git. Checkpoint training ở `runs/` và dữ liệu mặt/video ở `dataset/` cũng được loại khỏi Git.

Khi có dữ liệu cả team, train lại, calibration validation, export vào thư mục mới, kiểm tra và deploy bằng `scripts/deploy_model.py`. Chỉ chia sẻ release đã xác minh; cập nhật allowlist release trong `.gitignore` và symlink `active` cùng một commit. Các lớp, preprocessing và threshold phải đi cùng ONNX, tránh trộn bundle.

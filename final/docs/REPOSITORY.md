# Repository và Git LFS

EdgeFace nằm trong `final/` của repository `sonnguyen112/AIoT`. Mã nguồn, config mẫu, test, hướng dẫn và báo cáo kiến trúc/phương pháp được chia sẻ qua Git. Model ONNX đang dùng và detector YuNet được lưu bằng Git LFS.

## File được giữ cục bộ

- `.env`, bản backup `.env.*`, MQTT password file và SSH key.
- `dataset/`: video, crop khuôn mặt, ảnh Unknown, metadata và database thiết bị.
- `runs/`: checkpoint, lịch sử train/evaluation và hình diagnostic.
- `reports/generated/`: kết quả kiểm thử, telemetry, benchmark và state cục bộ.
- Cache pretrained, bản sao export và release rollback chưa được chọn để chia sẻ.
- `.venv*`, cache Python/pytest, metadata package và thư mục archive.

Việc dùng LFS không tự làm file riêng tư: dữ liệu tải qua LFS vẫn được người có quyền repository truy cập. Vì vậy dataset khuôn mặt và secrets không được đưa lên cả Git lẫn LFS.

## Clone

```bash
git lfs install
git clone https://github.com/sonnguyen112/AIoT.git
cd AIoT/final
git lfs pull
git lfs ls-files
```

Clone trên WSL/Linux để symlink `models/active` được tạo đúng. Dataset không có trong repository; có thể dùng model ONNX để demo hoặc chuẩn bị dataset riêng để train lại.

## Dọn file lỗi thời

Đã bỏ các script cài SSH tunnel/keepalive cũ, config tunnel sinh tự động và script tạo báo cáo/slide vốn cố định vào thí nghiệm `runs/mvp`. Tailscale là đường kết nối MQTT hiện tại. SSH quản trị và deployment vẫn hoạt động.

Báo cáo kết quả, hình benchmark và slide của thí nghiệm một thành viên được chuyển vào `.local_archive/cleanup-*/` trên máy phát triển, cùng manifest để khôi phục khi cần. Thư mục này không được upload. Dữ liệu/checkpoint/model rollback được giữ nguyên; runtime Docker/Pi không bị thay đổi bởi việc dọn repository.

Báo cáo hiện dùng: [ARCHITECTURE_AND_METHODS.md](../report/ARCHITECTURE_AND_METHODS.md). Đánh giá/benchmark phiên bản cả team và slide chính thức sẽ bổ sung sau khi có đủ dữ liệu.

## Chia sẻ model mới

Export, calibration và parity phải hoàn tất trước khi chọn release để chia sẻ. Cho phép đúng thư mục release trong `final/.gitignore`, cập nhật `models/active`, rồi kiểm tra:

```bash
git status --short
git check-attr filter -- models/releases/<release>/face_classifier.onnx
git lfs ls-files
git diff --cached --stat
```

File ONNX/checkpoint lớn có attribute LFS; ignore vẫn giữ checkpoint training riêng tư. Không dùng `git add -f dataset` hoặc `git add -f .env`. Git LFS phải được cài cả trên máy tạo commit lẫn máy clone.

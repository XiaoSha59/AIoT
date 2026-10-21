# Repository và Git LFS

EdgeFace nằm trong final/ của repository [sonnguyen112/AIoT](https://github.com/sonnguyen112/AIoT). Mã nguồn, cấu hình mẫu, tests, tài liệu vận hành và số liệu thực nghiệm không chứa ảnh mặt được chia sẻ qua Git.

## Tài liệu và kết quả đang dùng

- [PROJECT_HANDOFF.md](../report/PROJECT_HANDOFF.md): tài liệu tổng hợp cho team, làm nguồn viết báo cáo cuối kỳ và slide.
- [BENCHMARK_20261020.md](../report/BENCHMARK_20261020.md): phương pháp đo và đối sánh PyTorch/ONNX trên cùng Pi.
- report/experiments/team_20261020_son_live/: metadata training/evaluation/deployment của model hiện hành.
- report/experiments/benchmark_team_20261020/: số liệu benchmark và provenance.
- report/experiments/baseline_team_20261020.json: bằng chứng đối sánh head-only/fine-tune trước đợt cập nhật camera; dùng test 155 ảnh riêng, không trộn với test 181 ảnh hiện tại.
- report/assets/figures/: ảnh kiến trúc YuNet, MobileNetV3-Small và training loss, gồm PNG/SVG. Tạo lại bằng python -m scripts.render_report_figures.

## Model inference

Git LFS quản lý detector YuNet và các ONNX release được chọn. Giữ release 5010dc47644c đang active và 024f31fdc9b8 để rollback cùng classifier sáu lớp. Model demo hai lớp và export trùng đã được dọn. License và checksum của detector nằm trong models/.

Mỗi bundle gồm ONNX, classes, preprocessing, threshold và export validation. Chỉ đổi models/active sau khi export/checker/parity và kiểm tra trên Pi thành công. Không chỉnh riêng classes/threshold của một bundle đã xuất.

## Dữ liệu riêng trên máy phát triển

- .env, MQTT password file, SSH key và cấu hình tài khoản không đưa lên Git.
- Video gốc, phiên camera gốc của Sơn và dataset revision mới nhất 995 crop được giữ để train lại.
- Checkpoint tốt nhất của run hiện hành và checkpoint khởi tạo/head-only cần cho đối chứng được giữ ở runs/.
- File benchmark gốc của đợt hiện tại được giữ ở reports/generated/; bản metadata công bố nằm trong report/experiments/.
- .venv và dependency runtime đang sử dụng được giữ nguyên.
- Bản patch ghi lại chỉnh sửa thủ công trước đợt publish cũ được giữ riêng tại ~/.local/share/edgeface/backups/, ngoài repository.

Các dataset dẫn xuất lỗi thời (hai lớp, frame sai chiều, revision đã được bao hàm trong dataset mới), checkpoint trung gian, tar đồng bộ test, báo cáo trùng, kết quả smoke cũ và cache Python đã được dọn. Video gốc, ảnh camera gốc, secrets, cấu hình tên hiển thị và dịch vụ đang chạy không bị thay đổi.

LFS không làm file trở thành riêng tư: người có quyền repository vẫn tải được dữ liệu LFS. Không đưa ảnh khuôn mặt, checkpoint training hoặc secrets lên Git/LFS.

## Clone và làm việc

~~~bash
git lfs install
git clone https://github.com/sonnguyen112/AIoT.git
cd AIoT/final
git lfs pull
~~~

Clone trên WSL/Linux để symlink models/active được giữ đúng. Dataset/checkpoint riêng không có trong repository. Khi bổ sung release mới, cập nhật allowlist trong .gitignore cùng symlink active; kiểm tra git diff --cached và git lfs ls-files trước khi push.

Cleanup chỉ áp dụng cho final/. Các thay đổi và dữ liệu của midterm/ được giữ nguyên.

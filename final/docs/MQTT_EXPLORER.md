# Quan sát MQTT trên trình duyệt

Mở **http://localhost:4000** trên laptop. MQTT Explorer chạy bằng Docker Browser Mode từ [repository chính thức](https://github.com/thomasnordquist/MQTT-Explorer/blob/master/DOCKER.md), được ghim theo image digest.

- Đăng nhập bằng `MQTT_EXPLORER_USERNAME` và `MQTT_EXPLORER_PASSWORD` trong `.env` cục bộ. Không dùng Telegram token.
- Giao diện tự kết nối `mosquitto:1883` trong mạng Docker bằng tài khoản `edgeface_monitor`, chỉ được đọc `aiot/edgeface/#`.
- Mở nhánh `aiot → edgeface`: `status` là trạng thái mới nhất; `telemetry` cập nhật định kỳ; `command` xuất hiện khi gửi lệnh Telegram; `result` xuất hiện khi nhận diện.
- Gửi `/start`, `/status`, `/stop` từ Telegram để quan sát message. Command/result không retained, nên chỉ thấy các message phát sinh sau khi giao diện kết nối.
- Chọn topic để xem JSON, lịch sử và đồ thị các giá trị số. Không có ảnh/video trong MQTT.

Khởi động lại:

```bash
docker compose --profile bot --profile monitor up -d
```

Địa chỉ UI chỉ bind loopback laptop. Password UI và password MQTT là hai bộ khác nhau, đều ở `.env`; không gửi qua chat hoặc commit. Nếu cần đổi port, sửa `MQTT_EXPLORER_PORT` rồi chạy lại Compose.

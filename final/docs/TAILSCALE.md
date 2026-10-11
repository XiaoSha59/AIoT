# Kết nối WSL và Raspberry Pi bằng Tailscale

Tailscale được cài **native**, phiên bản thực tế `1.104.1`, trên WSL Ubuntu 24.04 và Pi Debian 12 ARM64. Hai máy đã đăng nhập cùng tài khoản/tailnet. Không dùng Docker cho Tailscale.

| Máy | Tên node | Tailscale IPv4 |
|---|---|---|
| Laptop WSL | `edgeface-wsl` | `100.73.104.13` |
| Raspberry Pi | `edgeface-pi` | `100.83.132.53` |

## MQTT hiện tại

```text
Pi: MQTT_HOST=100.73.104.13, port 1883
  → Tailscale VPN
  → WSL: Tailscale Serve TCP 1883
  → Mosquitto Docker: 127.0.0.1:1883
  ↔ Telegram bot / MQTT Explorer
```

Tailscale Serve chạy nền và lưu cấu hình trong daemon:

```bash
tailscale serve --bg --tcp=1883 tcp://127.0.0.1:1883
tailscale serve status
```

Chỉ các máy được policy tailnet cho phép mới truy cập được endpoint Tailscale. MQTT vẫn yêu cầu password và ACL theo role. Broker giữ loopback, không mở MQTT ra Wi-Fi hoặc Internet. Không dùng Funnel. SSH tunnel cũ đã dừng và disable sau khi Pi trả lời MQTT qua đường mới.

## Tự khởi động

- **Pi:** `tailscaled.service` enabled; tự chạy sau reboot, restart khi tiến trình thoát. `edgeface.service` cũng enabled, bắt đầu ở idle và reconnect MQTT.
- **WSL:** `/etc/wsl.conf` đã có `systemd=true`; `tailscaled.service` và Docker đều enabled. Drop-in `edgeface.conf` đặt `Restart=always`, `RestartSec=5`.
- **Windows:** tác vụ **EdgeFace WSL runtime**, chạy bằng user hiện tại, tự mở distro `Ubuntu` khi đăng nhập Windows và giữ `wsl.exe` hoạt động bằng `sleep infinity`. Tác vụ đang Running; không lưu mật khẩu Windows.

Systemd tự chạy dịch vụ khi WSL khởi động nhưng không tự giữ VM WSL sống. Tác vụ Windows giải quyết phần này. Laptop cần đang bật và có mạng; sleep/shutdown hoặc `wsl --shutdown` vẫn làm gián đoạn kết nối. Chưa thực hiện reboot Windows/Pi trong lần cấu hình này.

Kiểm tra ở WSL hoặc Pi:

```bash
systemctl is-enabled tailscaled
systemctl is-active tailscaled
tailscale ip -4
tailscale status
```

Kiểm tra tác vụ từ PowerShell:

```powershell
Get-ScheduledTask -TaskName 'EdgeFace WSL runtime'
```

## Cài lại trên máy tương thích

Chạy tại terminal, nhập mật khẩu sudo nếu được yêu cầu:

```bash
bash scripts/setup_tailscale.sh edgeface-wsl
tailscale up --hostname=edgeface-wsl --accept-dns=false --accept-routes=false
```

Trên Pi dùng `edgeface-pi` thay cho `edgeface-wsl`. Script cài từ repository stable chính thức, bật systemd và cấp quyền operator cho user hiện tại. Mở link đăng nhập trong trình duyệt bằng cùng tài khoản. Không đặt auth key hoặc mật khẩu trong source/.env.

Để cài tác vụ tự mở WSL, chạy [install_wsl_autostart.ps1](../scripts/install_wsl_autostart.ps1) từ PowerShell, truyền tên distro ở `wsl -l -q`:

```powershell
# Chạy từ thư mục final/scripts đã clone, điều chỉnh tên distro theo máy.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install_wsl_autostart.ps1 -Distribution Ubuntu
```

Sau khi xác minh SSH host key, chuyển MQTT bằng script (preserve các config/credentials khác, rollback nếu không có ack):

```bash
python -m scripts.connect_tailscale --pi-host pi@<PI_SSH_IP>
```

Địa chỉ Tailscale giữ được khi đổi Wi-Fi nếu vẫn giữ node/state hiện có và hai máy truy cập Internet. Pi vẫn phải có cấu hình Wi-Fi/IP vật lý phù hợp: profile IP tĩnh LAN cũ chỉ dùng trên Wi-Fi hiện tại; mạng khác nên tạo profile DHCP. Không cần đổi `MQTT_HOST` mỗi lần đổi mạng.

## Kiểm chứng và xử lý lỗi

Đã đo ping hai chiều qua đường trực tiếp khoảng 10–11 ms, SSH qua Tailscale và kết nối TCP Pi → WSL:1883. Artifact MQTT có request ID mới và kiểm thử camera nằm ở `reports/generated/tailscale_network.json` / `tailscale_mqtt_smoke.json`; UI nhận status/telemetry thật. Auto-stop của bot đã được kiểm thử với nhận diện thật; restart dịch vụ Tailscale Pi xong MQTT vẫn phản hồi. SSH host key đã pin cho IP Tailscale Pi, `PI_HOST` cục bộ đã cập nhật. Số đo này thuộc mạng hiện tại, không bảo đảm latency khi đổi mạng.

```bash
tailscale ping --c 3 --timeout 5s 100.83.132.53
python -m scripts.mqtt_command status
python -m scripts.explorer_smoke
```

Nếu node hết hạn key, đăng nhập lại bằng `tailscale up`. Policy tailnet cần cho Pi kết nối WSL TCP 1883; SSH cần cho WSL kết nối Pi TCP 22. Không thay đổi policy của các thiết bị khác trong tài khoản. Nếu đường mạng bị relay hoặc MTU bất thường, kiểm tra `tailscale status`/`tailscale ping`; hiện kết nối đã được kiểm chứng mặc dù Windows cũng có Tailscale đang chạy.

Nguồn: [Tailscale WSL2](https://tailscale.com/docs/install/windows/wsl2), [Linux native](https://tailscale.com/docs/install/linux), [Serve TCP](https://tailscale.com/docs/reference/tailscale-cli/serve), [Microsoft: systemd không giữ WSL sống](https://learn.microsoft.com/en-us/windows/wsl/systemd).

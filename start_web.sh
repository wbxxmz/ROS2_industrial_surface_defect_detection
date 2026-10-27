#!/usr/bin/env bash
# 扩展 04 §3.1：Linux 双击启动 Web 服务（自检 → 选配置 → 起服务 → 开浏览器）
#
# 用法：
#   ./start_web.sh               # 启动服务并自动打开浏览器；Ctrl+C 停止
#   ./start_web.sh --no-browser  # 只启动服务（远程 / 无桌面环境）
#
# 退出码：0=正常退出；2=环境自检失败；3=服务启动超时或异常退出
#
# 说明：
#   - 依赖自检只覆盖 requirements.txt 内的 Web 依赖；
#     torch / ultralytics 不在该清单内（见 README 安装步骤第 3 步）。
#   - 若已设置 DEFECT_WEB_CONFIG，则优先使用该配置，跳过自动选择。

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

OPEN_BROWSER=1
for arg in "$@"; do
  case "$arg" in
    --no-browser) OPEN_BROWSER=0 ;;
    -h|--help)
      sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "[错误] 未知参数：$arg（可用：--no-browser）" >&2
      exit 2
      ;;
  esac
done

fail() {
  echo "[错误] $1" >&2
  exit 2
}

# ---------- 1. Python 版本 ----------
if ! command -v python3 >/dev/null 2>&1; then
  fail "未找到 python3，请先安装 Python 3.10+（如 sudo apt install python3）"
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
  fail "Python 版本过低（当前 $(python3 -V 2>&1)），需要 3.10 及以上"
fi

# ---------- 2. 依赖自检（仅 requirements.txt 内 Web 依赖） ----------
MISSING="$(python3 - <<'PY'
import importlib.util

modules = {
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "yaml": "PyYAML",
    "reportlab": "reportlab",
    "openpyxl": "openpyxl",
    "matplotlib": "matplotlib",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
}
missing = [pkg for mod, pkg in modules.items() if importlib.util.find_spec(mod) is None]
# python-multipart 0.0.12+ 同时提供 multipart / python_multipart 两个模块名
if importlib.util.find_spec("multipart") is None and importlib.util.find_spec("python_multipart") is None:
    missing.append("python-multipart")
print(" ".join(missing))
PY
)" || fail "依赖自检失败：无法启动 python3 解释器"

if [ -n "$MISSING" ]; then
  echo "[错误] 缺少依赖：$MISSING" >&2
  echo "        请执行：pip install -r requirements.txt" >&2
  exit 2
fi

# ---------- 3. 配置选择 ----------
if [ -n "${DEFECT_WEB_CONFIG:-}" ]; then
  CONFIG="$DEFECT_WEB_CONFIG"
  echo "[信息] 使用环境变量 DEFECT_WEB_CONFIG：$CONFIG"
elif [ -f "best.pt" ]; then
  if python3 -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("ultralytics") else 1)' >/dev/null 2>&1; then
    CONFIG="configs/html_yolo_demo.yaml"
    echo "[信息] 检测到 best.pt + ultralytics → 使用 YOLO 路线"
  else
    CONFIG="configs/html_demo.yaml"
    echo "[提示] 检测到 best.pt，但未安装 ultralytics / torch → 回退 OpenCV 路线"
    echo "        如需 YOLO：pip install torch ultralytics（有 GPU 时按官网选 CUDA 轮子）"
  fi
else
  CONFIG="configs/html_demo.yaml"
  echo "[提示] 未找到 best.pt（约 52 MB，不随仓库分发）→ 使用 OpenCV 路线"
fi

if [ ! -f "$CONFIG" ]; then
  fail "配置文件不存在：$CONFIG"
fi

# ---------- 4. 端口探测（8000 → 8001 → 8002） ----------
PORT="$(python3 - <<'PY'
import socket

for port in (8000, 8001, 8002):
    with socket.socket() as sock:
        # 允许复用 TIME_WAIT 端口：否则刚 Ctrl+C 再启动会跳到 8001，体验上像"端口没释放"
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    print("")
PY
)" || fail "端口探测失败：无法启动 python3 解释器"
if [ -z "$PORT" ]; then
  fail "端口 8000–8002 均被占用，请释放其中之一后重试"
fi

URL="http://127.0.0.1:$PORT"
LOG="reports/web/server.log"
mkdir -p reports/web

echo "[信息] 项目根目录：$PROJECT_ROOT"
echo "[信息] 配置文件：$CONFIG"
echo "[信息] 访问地址：$URL"
echo "[信息] 服务日志：$PROJECT_ROOT/$LOG"

# ---------- 5. 启动服务 + 健康检查（最多 30 s） ----------
DEFECT_WEB_CONFIG="$PROJECT_ROOT/$CONFIG" \
  python3 -m uvicorn web.app:create_app --factory --host 127.0.0.1 --port "$PORT" \
  >>"$LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    echo "[信息] 服务已停止，端口 $PORT 已释放"
  fi
}
trap cleanup INT TERM EXIT

READY=0
for _ in $(seq 1 30); do
  if python3 - "$URL" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

urllib.request.urlopen(sys.argv[1] + "/", timeout=1).read(64)
PY
  then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "[错误] 服务未能在 30 秒内就绪，日志末尾：" >&2
  tail -n 20 "$LOG" >&2 || true
  exit 3
fi

echo "[完成] 服务已就绪：$URL"

if [ "$OPEN_BROWSER" -eq 1 ] && { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; } \
   && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 &
fi

echo "[提示] 在浏览器中拖入图片即可检测（.jpg / .jpeg / .png / .bmp，≤ 10 MB）"
echo "[提示] 按 Ctrl+C 停止服务"

wait "$SERVER_PID"
STATUS=$?
if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 130 ]; then
  echo "[错误] 服务异常退出（退出码 $STATUS），日志末尾：" >&2
  tail -n 20 "$LOG" >&2 || true
  exit 3
fi
exit 0

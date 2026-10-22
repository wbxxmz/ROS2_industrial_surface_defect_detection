#!/usr/bin/env bash
# 阶段 9 §4.2 / §5：一键启动脚本（环境自检 + 可读报错，不打印裸 traceback）
#
# 用法：
#   ./run_demo.sh              # 默认 OpenCV 路线
#   ./run_demo.sh opencv       # OpenCV 传统算法
#   ./run_demo.sh yolo         # YOLOv8m 路线（需 best.pt）
#   ./run_demo.sh benchmark    # 批量压测（需 psutil）
#
# 退出码：0=跑通并生成报告；2=环境自检失败；1=业务运行失败

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-opencv}"
case "$MODE" in
  opencv|yolo|benchmark) ;;
  *)
    echo "[错误] 未知模式：$MODE" >&2
    echo "用法：./run_demo.sh [opencv|yolo|benchmark]（默认 opencv）" >&2
    exit 2
    ;;
esac

fail() {
  echo "[错误] $1" >&2
  exit 2
}

# ---------- 1. Python 版本 ----------
if ! command -v python3 >/dev/null 2>&1; then
  fail "未找到 python3，请先安装 Python 3.10+（如 sudo apt install python3）"
fi

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
  PY_VER="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || echo "未知")"
  fail "Python 版本过低（当前 $PY_VER），需要 3.10 及以上"
fi

# ---------- 2. 核心依赖 ----------
MISSING="$(python3 - <<'PY'
import importlib.util

modules = {
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "reportlab": "reportlab",
    "openpyxl": "openpyxl",
    "matplotlib": "matplotlib",
    "psutil": "psutil",
}
missing = [pkg for mod, pkg in modules.items() if importlib.util.find_spec(mod) is None]
print(" ".join(missing))
PY
)" || fail "依赖自检失败：无法启动 python3 解释器"

if [ -n "$MISSING" ]; then
  echo "[错误] 缺少核心依赖：$MISSING" >&2
  echo "        请执行：pip install -r requirements.txt" >&2
  exit 2
fi

# ---------- 3. YOLO 模式额外检查 ----------
if [ "$MODE" = "yolo" ]; then
  if ! python3 -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("ultralytics") else 1)' >/dev/null 2>&1; then
    echo "[错误] 缺少 YOLO 依赖：ultralytics / torch" >&2
    echo "        请执行：pip install ultralytics torch" >&2
    echo "        或改用 OpenCV 路线：./run_demo.sh opencv" >&2
    exit 2
  fi
  if [ ! -f "best.pt" ]; then
    echo "[错误] 未找到模型权重 best.pt（约 52 MB，未随仓库分发）" >&2
    echo "        请先执行：python3 train_yolo.py" >&2
    echo "        或从 README「模型权重获取方式」指定地址下载后放到项目根目录" >&2
    exit 2
  fi
fi

# ---------- 4. 样例图片 ----------
if [ ! -d "test_picture" ] || [ -z "$(find test_picture -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' \) -print -quit)" ]; then
  fail "样例目录 test_picture/ 不存在或为空，请先放入至少一张图片"
fi

# ---------- 5. 按模式执行 ----------
echo "[信息] 模式：$MODE；项目根目录：$PROJECT_ROOT"

case "$MODE" in
  opencv)
    python3 main.py --config configs/demo.yaml
    STATUS=$?
    REPORT="reports/quality_report.pdf"
    ;;
  yolo)
    python3 main.py --config configs/yolo_demo.yaml
    STATUS=$?
    REPORT="reports/quality_report_yolo.pdf"
    ;;
  benchmark)
    if [ -d "tests/data/stress_images" ]; then
      INPUT="tests/data/stress_images"
      LIMIT=30
    else
      INPUT="test_picture"
      LIMIT=2
      echo "[提示] 未找到压测素材 tests/data/stress_images，改用 test_picture/ 作为输入"
    fi
    python3 scripts/benchmark.py --config configs/yolo_demo.yaml --input "$INPUT" --limit "$LIMIT"
    STATUS=$?
    REPORT="reports/benchmark_result.json"
    ;;
esac

if [ "$STATUS" -ne 0 ]; then
  echo "[错误] $MODE 模式运行失败（退出码 $STATUS），详见上方日志" >&2
  exit 1
fi

if [ -f "$REPORT" ]; then
  echo "[完成] 报告已生成：$PROJECT_ROOT/$REPORT"
else
  echo "[完成] 运行成功，但未找到预期报告文件：$REPORT" >&2
fi
exit 0

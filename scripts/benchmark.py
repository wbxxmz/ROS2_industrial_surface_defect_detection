"""阶段 8 §4.2 / §5：批量压测脚本（异常隔离 + 性能基线）。

按阶段 8 文档的接口签名实现：
    python3 scripts/benchmark.py --config configs/yolo_demo.yaml \
        --input tests/data/stress_images --limit 30 \
        --output reports/benchmark_result.json

做的事：
1. 逐图/逐帧走完整链路 detect → grade → judge → render，统计耗时与内存；
2. 单张图片损坏、目录为空等异常**逐项捕获并记账**，不中断整批（阶段 8 §4.4）；
3. 输出 JSON 指标报告（总耗时、平均/中位/最大/最小单帧耗时、检测 FPS、峰值 RSS、
   成功/失败帧数、原始检出与有效缺陷数、错误清单）。

说明：图片目录按单文件逐个读取（每个文件独立 ``ImageSource``），这样才能在某个文件
解码失败时跳过并继续下一张——``ImageSource`` 目录模式在 ``read()`` 失败时不推进游标，
继续调用只会重复同一错误。
"""

import argparse
import json
import logging
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inputs.factory import build_source  # noqa: E402
from inputs.image_source import ImageSource  # noqa: E402
from judge import judge  # noqa: E402
from pipeline.pipeline import VISUAL_OUTPUT_DIR, build_detector  # noqa: E402
from severity import grade  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.exceptions import ProjectError  # noqa: E402
from visualization import render_to_file  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_LIMIT = 30
DEFAULT_OUTPUT = "reports/benchmark_result.json"

logger = logging.getLogger("defect.benchmark")


# --------------------------------------------------------------------------- #
# 峰值内存采样：后台线程按固定间隔读 RSS，规避"只在帧边界采样会低估瞬态峰值"
# --------------------------------------------------------------------------- #
class PeakRSS:
    """后台采样进程 RSS，返回峰值（MB）；未安装 psutil 时返回 None。"""

    def __init__(self, interval: float = 0.02) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._peak = 0
        self._available = True
        try:
            import psutil

            self._process = psutil.Process()
        except Exception:  # psutil 缺失或平台不支持：降级为不统计
            self._available = False
            self._process = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._peak = max(self._peak, self._process.memory_info().rss)
            except Exception:  # 进程退出瞬间读取失败：忽略
                break

    def start(self) -> None:
        if not self._available:
            return
        self._peak = self._process.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if not self._available:
            return None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self._peak / (1024 * 1024)


# --------------------------------------------------------------------------- #
# 输入解析
# --------------------------------------------------------------------------- #
def resolve_path(value: str) -> Path:
    """相对路径先按当前目录，再按项目根目录解析。"""
    path = Path(value)
    if path.exists():
        return path.resolve()
    candidate = PROJECT_ROOT / path
    return candidate.resolve() if candidate.exists() else path.resolve()


def classify_input(path: Path) -> str:
    """判定输入类别：directory / video / image。"""
    if path.is_dir():
        return "directory"
    if path.suffix.lower() in VIDEO_EXTS:
        return "video"
    return "image"


def collect_images(directory: Path, extensions: list[str]) -> list[Path]:
    """按扩展名白名单收集目录内图片，文件名排序（与 ImageSource 目录模式一致）。"""
    allowed = {ext.lower() for ext in extensions}
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in allowed)


# --------------------------------------------------------------------------- #
# 指标统计
# --------------------------------------------------------------------------- #
def summarize(values: list[float]) -> dict:
    """耗时统计（ms）；空列表返回全 0，避免除零。"""
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "p95": round(ordered[index], 2),
    }


def build_input_cfg(config: dict, input_path: Path, kind: str) -> dict:
    """构造输入源配置：命令行 --input 覆盖配置文件中的 input 段。"""
    base = dict(config.get("input") or {})
    extensions = base.get("extensions") or sorted(IMAGE_EXTS)
    if kind == "directory":
        return {"type": "image", "dir": str(input_path), "extensions": list(extensions)}
    if kind == "video":
        return {"type": "video", "path": str(input_path)}
    return {"type": "image", "path": str(input_path)}


def environment_info(config: dict) -> dict:
    """记录运行环境，避免 CPU / GPU 指标混比（阶段 8 §4.2）。"""
    import os

    info = {"cpu_count": os.cpu_count(), "device": (config.get("detector") or {}).get("params", {}).get("device")}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # 未安装 torch：纯 OpenCV 路线
        info["torch"] = None
        info["cuda_available"] = False
    return info


# --------------------------------------------------------------------------- #
# 压测主流程
# --------------------------------------------------------------------------- #
def run_benchmark(config: dict, input_path: Path, limit: int) -> dict:
    """执行压测并返回指标字典；单个输入项失败只记账不中断。"""
    kind = classify_input(input_path)
    input_cfg = build_input_cfg(config, input_path, kind)

    detector = build_detector(config)
    load_start = time.perf_counter()
    detector.load(config.get("detector") or {})
    load_ms = (time.perf_counter() - load_start) * 1000.0

    peak = PeakRSS()
    detect_ms: list[float] = []
    frame_ms: list[float] = []
    errors: list[dict] = []
    total_detections = 0
    total_valid = 0
    success = 0
    requested = 0
    warmed = False

    def record_error(index: int, name: str, exc: Exception) -> None:
        errors.append(
            {
                "index": index,
                "source": name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
        logger.warning("样本 %d（%s）处理失败：%s: %s", index, name, type(exc).__name__, exc)

    peak.start()
    loop_start = time.perf_counter()
    try:
        if kind == "video":
            source = build_source(input_cfg)
            try:
                source.open(input_cfg)
            except ProjectError as exc:  # 视频打不开：记账后收尾，不中断
                requested += 1
                record_error(1, input_path.name, exc)
                source = None
            if source is not None:
                try:
                    index = 0
                    while index < limit:
                        try:
                            frame = source.read()
                        except ProjectError as exc:  # 中途解码失败：该帧记账后收尾
                            record_error(index, str(input_path), exc)
                            break
                        if frame is None:
                            break
                        index += 1
                        requested += 1
                        name = f"{input_path.stem}_frame{frame.frame_index:06d}"
                        try:
                            if not warmed:  # 首次推理含 CUDA 初始化/算子自调优，不计入计时
                                detector.detect(frame)
                                warmed = True
                            start = time.perf_counter()
                            det_start = time.perf_counter()
                            detections = grade(detector.detect(frame), config.get("severity"))
                            detect_cost = (time.perf_counter() - det_start) * 1000.0
                            verdict = judge(detections, config.get("judge"))
                            render_to_file(
                                frame, detections, verdict, str(VISUAL_OUTPUT_DIR / f"{name}.jpg")
                            )
                            frame_ms.append((time.perf_counter() - start) * 1000.0)
                            detect_ms.append(detect_cost)
                            success += 1
                            total_detections += len(detections)
                            total_valid += verdict.total_count
                        except ProjectError as exc:
                            record_error(index, name, exc)
                finally:
                    source.close()
        else:
            paths = [input_path] if kind == "image" else collect_images(input_path, input_cfg["extensions"])
            if kind == "directory":
                # 用目录模式走一次 open：空目录/不存在按契约抛 SourceOpenError，此处隔离记账
                probe = build_source(input_cfg)
                try:
                    probe.open(input_cfg)
                    probe.close()
                except ProjectError as exc:
                    requested += 1
                    record_error(1, str(input_path), exc)
                    paths = []
            for index, path in enumerate(paths[:limit], start=1):
                requested += 1
                source = ImageSource()
                try:
                    source.open({"type": "image", "path": str(path)})
                    frame = source.read()
                    if frame is None:  # 契约上 open 已校验，理论上不会发生
                        raise ProjectError(f"图片读取为空：{path}")
                    if not warmed:  # 首次推理含 CUDA 初始化/算子自调优，不计入计时
                        detector.detect(frame)
                        warmed = True
                    start = time.perf_counter()
                    det_start = time.perf_counter()
                    detections = grade(detector.detect(frame), config.get("severity"))
                    detect_ms.append((time.perf_counter() - det_start) * 1000.0)
                    verdict = judge(detections, config.get("judge"))
                    render_to_file(
                        frame, detections, verdict, str(VISUAL_OUTPUT_DIR / f"{path.stem}.jpg")
                    )
                    frame_ms.append((time.perf_counter() - start) * 1000.0)
                    success += 1
                    total_detections += len(detections)
                    total_valid += verdict.total_count
                except ProjectError as exc:  # 损坏图片 / 单文件打开失败：隔离并继续
                    record_error(index, path.name, exc)
                except Exception as exc:  # 非契约异常也不允许打断整批
                    record_error(index, path.name, exc)
                finally:
                    source.close()
    finally:
        peak_rss_mb = peak.stop()
        detector.release()

    total_elapsed = time.perf_counter() - loop_start
    timing = summarize(frame_ms)
    detect_timing = summarize(detect_ms)
    fps_detect = round(1000.0 / detect_timing["mean"], 2) if detect_timing["mean"] else 0.0
    fps_e2e = round(1000.0 / timing["mean"], 2) if timing["mean"] else 0.0

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": None,  # 由 main 回填
        "input": str(input_path),
        "input_kind": kind,
        "limit": limit,
        "environment": environment_info(config),
        "load_ms": round(load_ms, 2),
        "total_elapsed_s": round(total_elapsed, 3),
        "frames": {"requested": requested, "success": success, "error": len(errors)},
        "frame_timing_ms": timing,
        "detect_timing_ms": detect_timing,
        "fps": {"detect": fps_detect, "end_to_end": fps_e2e},
        "peak_rss_mb": round(peak_rss_mb, 2) if peak_rss_mb is not None else None,
        "detections": {"raw": total_detections, "valid": total_valid},
        "errors": errors,
        "error_count": len(errors),
    }


def print_summary(metrics: dict) -> None:
    """按阶段 8 §4.2 的指标口径打印终端摘要。"""
    frames = metrics["frames"]
    timing = metrics["frame_timing_ms"]
    detect = metrics["detect_timing_ms"]
    print(f"[压测] 输入：{metrics['input']}（{metrics['input_kind']}，上限 {metrics['limit']}）")
    print(f"[压测] 成功帧数：{frames['success']}  失败帧数：{frames['error']}")
    print(
        f"[压测] 检测耗时：平均 {detect['mean']:.2f} ms，中位 {detect['median']:.2f} ms，"
        f"最小 {detect['min']:.2f} ms，最大 {detect['max']:.2f} ms → {metrics['fps']['detect']:.2f} FPS"
    )
    print(
        f"[压测] 端到端：平均 {timing['mean']:.2f} ms，中位 {timing['median']:.2f} ms，"
        f"P95 {timing['p95']:.2f} ms → {metrics['fps']['end_to_end']:.2f} FPS"
    )
    print(f"[压测] 总耗时：{metrics['total_elapsed_s']:.3f} s（模型加载 {metrics['load_ms']:.1f} ms 不计入）")
    mem = metrics["peak_rss_mb"]
    print(f"[压测] 峰值内存：{mem:.1f} MB" if mem is not None else "[压测] 峰值内存：psutil 不可用")
    print(
        f"[压测] 检出缺陷：原始 {metrics['detections']['raw']} 条，"
        f"有效 {metrics['detections']['valid']} 条"
    )
    for error in metrics["errors"]:
        print(f"[错误] #{error['index']} {error['source']} → {error['error_type']}: {error['message']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="阶段 8 批量压测（异常隔离 + 性能基线）")
    parser.add_argument("--config", default="configs/demo.yaml", help="算法配置文件")
    parser.add_argument("--input", default=None, help="待测文件或目录（缺省取配置文件 input 段）")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"最大样本数，默认 {DEFAULT_LIMIT}")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"JSON 指标输出路径，默认 {DEFAULT_OUTPUT}")
    args = parser.parse_args(argv)

    if args.limit <= 0:
        print("[错误] --limit 必须为正整数", file=sys.stderr)
        return 1

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    config_path = resolve_path(args.config)
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] 配置加载失败：{exc}", file=sys.stderr)
        return 1

    input_path = resolve_path(args.input) if args.input else None
    if input_path is None:
        input_cfg = config.get("input") or {}
        raw = input_cfg.get("path") or input_cfg.get("dir")
        if not raw:
            print("[错误] 未指定 --input，且配置文件缺少 input.path / input.dir", file=sys.stderr)
            return 1
        input_path = resolve_path(str(raw))

    try:
        metrics = run_benchmark(config, input_path, args.limit)
    except (ProjectError, ValueError) as exc:  # 模型加载失败 / 配置非法：无法启动，非零退出
        print(f"[错误] 无法启动压测：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    metrics["config"] = str(config_path)
    print_summary(metrics)

    output_path = resolve_path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[错误] 指标写入失败：{exc}", file=sys.stderr)
        return 1
    print(f"[压测] 结果已写入：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

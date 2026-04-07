"""
EyeTrace 项目总控脚本
====================

功能：
- 从项目根目录启动整个系统；
- 一键启动【数据采集】（Cognitive 目录里的 aoi_collector_v3_2.py）；
- 可选：在单独的进程里启动【准实时监控+预测】（Model/ET_model/realtime_session_monitor.py）。

使用方式（推荐在项目根目录运行）::

    cd C:\\Users\\YNS\\Desktop\\EyeTrace
    py -3.10 EyeTrace_controller.py           # 只做数据采集
    py -3.10 EyeTrace_controller.py --with-monitor  # 采集 + 启动准实时预测监控
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent  # 项目根目录：C:\\Users\\YNS\\Desktop\\EyeTrace


def run_collector() -> int:
    """
    调用 Cognitive 里的采集脚本 aoi_collector_v3_2.py。

    说明：
    - 不改你原来的采集逻辑，只是统一从根目录启动；
    - 使用绝对路径调用，避免当前工作目录不同导致的导入/路径问题。
    """
    collector_path = ROOT / "Cognitive" / "cognitive-load-tracker" / "cognitive_study" / "aoi_collector_v3_2.py"
    if not collector_path.exists():
        print(f"[Controller] 找不到采集脚本：{collector_path}")
        return 1

    cmd = [sys.executable, str(collector_path)]
    print(f"[Controller] 启动采集脚本：{' '.join(cmd)}")
    # 直接把采集脚本作为前台进程运行，让你能看到原有的所有输出和界面
    return subprocess.call(cmd, cwd=str(ROOT))


def run_monitor(background: bool = True) -> subprocess.Popen | int:
    """
    启动准实时监控/预测脚本 `Model/ET_model/realtime_session_monitor.py`（只负责“算”）。

    默认参数：
    - --watch_dirs Cognitive/data/cognitive_study data
    - --interval 10
    这些都和你在 realtime_session_monitor.py 里写的推荐用法一致。
    """
    monitor_path = ROOT / "Model" / "ET_model" / "realtime_session_monitor.py"
    if not monitor_path.exists():
        print(f"[Controller] 找不到监控脚本：{monitor_path}")
        return 1

    cmd = [
        sys.executable,
        str(monitor_path),
        "--watch_dirs",
        "Cognitive/data/cognitive_study",
        "data",
        "--interval",
        "10",
    ]
    print(f"[Controller] 启动准实时监控脚本：{' '.join(cmd)}")

    if background:
        # 作为后台进程跑，只在控制台输出少量信息，不影响采集窗口
        return subprocess.Popen(cmd, cwd=str(ROOT))
    else:
        return subprocess.call(cmd, cwd=str(ROOT))


def run_realtime_dashboard(background: bool = True) -> subprocess.Popen | int:
    """
    启动实时预测可视化面板 `Model/ET_model/realtime_dashboard.py`（负责“看”）。

    注意：
    - 该脚本会读取 `Model/ET_model/realtime_predictions_task_supervised.jsonl`，
      所以需要与 `realtime_session_monitor.py` 一起运行才有数据源。
    """
    dashboard_path = ROOT / "Model" / "ET_model" / "realtime_dashboard.py"
    if not dashboard_path.exists():
        print(f"[Controller] 找不到实时面板脚本：{dashboard_path}")
        return 1

    cmd = [
        sys.executable,
        str(dashboard_path),
    ]
    print(f"[Controller] 启动实时预测面板脚本：{' '.join(cmd)}")

    if background:
        # 作为后台进程跑，在单独的窗口里显示 Tkinter 面板，不阻塞采集主进程
        return subprocess.Popen(cmd, cwd=str(ROOT))
    else:
        return subprocess.call(cmd, cwd=str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EyeTrace 项目总控脚本（统一从根目录启动采集与预测）。")
    parser.add_argument(
        "--with-monitor",
        action="store_true",
        help="在启动采集的同时，额外启动一套“准实时预测”：监控 + 实时面板。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    monitor_proc = None
    dashboard_proc = None
    if args.with_monitor:
        monitor_proc = run_monitor(background=True)
        if isinstance(monitor_proc, int) and monitor_proc != 0:
            print("[Controller] 准实时监控脚本启动失败，但采集仍会继续。")
            monitor_proc = None

        # 只有当监控脚本成功启动时，才去拉起实时面板
        if monitor_proc is not None:
            dashboard_proc = run_realtime_dashboard(background=True)
            if isinstance(dashboard_proc, int) and dashboard_proc != 0:
                print("[Controller] 实时预测面板启动失败，但采集和监控仍会继续。")
                dashboard_proc = None

    exit_code = run_collector()

    # 如果你希望在采集结束时顺便结束监控进程，可以在这里做清理
    if monitor_proc is not None and hasattr(monitor_proc, "terminate"):
        try:
            monitor_proc.terminate()
        except Exception:
            pass

    if dashboard_proc is not None and hasattr(dashboard_proc, "terminate"):
        try:
            dashboard_proc.terminate()
        except Exception:
            pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()


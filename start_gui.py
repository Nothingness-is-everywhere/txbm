"""
GUI启动脚本
用于启动 ok-script 的图形界面
"""
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

from ok import OK

config = {
    "debug": True,
    "use_gui": True,
    "gui_title": "ok-script",
    "gui_icon": ":/icon/icon.ico",
    "version": "1.0.0",
    "window_size": {
        "width": 1000,
        "height": 800,
        "min_width": 600,
        "min_height": 450,
    },
    "adb": {
        "packages": ["com.pinkcore.tkfm"],
    },
    "onetime_tasks": [
        ["ok.automation.game_startup_task", "GameStartupTask"],
    ],
    "trigger_tasks": [
        ["ok_tasks.NetworkErrorHandler", "NetworkErrorHandler"],
    ],
}

if __name__ == "__main__":
    print("正在启动 ok-script GUI...")
    print(f"Python: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    try:
        ok = OK(config)
        ok.start()
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

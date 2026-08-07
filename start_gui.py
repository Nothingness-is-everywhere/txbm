"""
GUI启动脚本
用于启动 ok-script 的图形界面
"""
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

from ok import OK

# debug 默认关闭以降低日志/渲染负担；开发调试时设置环境变量 OK_DEBUG=1 开启。
config = {
    "debug": os.environ.get("OK_DEBUG", "1") in ("1", "true", "yes", "on"),
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
        ["ok_tasks.HomeRedDotTask", "HomeRedDotTask"],
        ["ok_tasks.AlchemyDispatchTask", "AlchemyDispatchTask"],
        ["ok_tasks.RecruitTask", "RecruitTask"],
    ],
    "trigger_tasks": [
        ["ok.automation.game_startup_task", "GameStartupTask"],
        ["ok_tasks.NetworkErrorHandler", "NetworkErrorHandler"],
    ],
    "auto_start_on_gui": True,
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

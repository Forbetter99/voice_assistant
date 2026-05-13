import datetime
import logging
import os
import subprocess
import webbrowser

logger = logging.getLogger(__name__)

APP_ALIASES = {
    "计算器": "calc.exe",
    "记事本": "notepad.exe",
    "画图": "mspaint.exe",
    "浏览器": None,
    "设置": "ms-settings:",
    "任务管理器": "taskmgr.exe",
    "命令提示符": "cmd.exe",
    "powershell": "powershell.exe",
    "控制面板": "control",
    "文件资源管理器": "explorer.exe",
    "截图工具": "SnippingTool.exe",
}


class IntentHandler:
    def execute(self, intent, entities, config):
        handler_map = {
            "open_app": self._handle_open_app,
            "search": self._handle_search,
            "time_query": self._handle_time_query,
            "weather": self._handle_weather,
            "system_cmd": self._handle_system_cmd,
            "chat": self._handle_chat,
        }
        handler = handler_map.get(intent, self._handle_chat)
        return handler(entities, config)

    def _handle_open_app(self, entities, config):
        app_name = entities.get("app_name", "")
        if not app_name:
            return "请告诉我你要打开什么应用"

        alias = APP_ALIASES.get(app_name)
        if alias is None and app_name in APP_ALIASES:
            webbrowser.open("https://www.baidu.com")
            return f"已打开浏览器"
        elif alias:
            try:
                if alias.startswith("ms-"):
                    subprocess.Popen(["start", alias], shell=True)
                else:
                    subprocess.Popen(alias)
                return f"正在打开{app_name}"
            except Exception as e:
                logger.error(f"Failed to open {app_name}: {e}")
                return f"打开{app_name}失败"

        try:
            subprocess.Popen(f"start {app_name}", shell=True)
            return f"正在尝试打开{app_name}"
        except Exception:
            webbrowser.open(f"https://www.baidu.com/s?wd={app_name}")
            return f"已为你搜索{app_name}相关信息"

    def _handle_search(self, entities, config):
        query = entities.get("query", "")
        if not query:
            return "请问你要搜索什么"
        webbrowser.open(f"https://www.baidu.com/s?wd={query}")
        return f"正在搜索{query}"

    def _handle_time_query(self, entities, config):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y年%m月%d日")
        time_str = now.strftime("%H点%M分")
        weekday = now.weekday()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return f"今天是{date_str}，{weekdays[weekday]}，当前时间{time_str}"

    def _handle_weather(self, entities, config):
        location = entities.get("location", "本地")
        return f"抱歉，天气查询功能需要配置天气API，{location}的天气暂时无法查询"

    def _handle_system_cmd(self, entities, config):
        command = entities.get("command", "")
        if command == "shutdown":
            os.system("shutdown /s /t 5")
            return "系统将在5秒后关机"
        elif command == "sleep":
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
            return "系统即将进入睡眠状态"
        elif command == "volume_up":
            os.system("nircmd changesysvolume 2000")
            return "已增大音量"
        elif command == "volume_down":
            os.system("nircmd changesysvolume -2000")
            return "已减小音量"
        elif command == "mute":
            os.system("nircmd mutesysvolume 1")
            return "已静音"
        elif command == "screenshot":
            os.system("SnippingTool.exe")
            return "正在打开截图工具"
        return f"未知系统命令: {command}"

    @staticmethod
    def _handle_chat(entities, config):
        response = entities.get("response", "")
        return response if response else "嗯，我在听"

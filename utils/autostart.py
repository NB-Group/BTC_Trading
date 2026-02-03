import os
import subprocess
from btc_predictor.utils import LOGGER
import config


def _quote(arg: str) -> str:
    if not arg:
        return ""
    if ' ' in arg or '"' in arg:
        escaped = arg.replace('"', r'\"')
        return f'"{escaped}"'
    return arg


def ensure_windows_autostart():
    """在 Windows 上通过 schtasks 注册“登录即启动”计划任务。
    - 任务名来自 config.AUTO_START.task_name
    - 程序路径为当前 Python 解释器 + 项目 main.py
    - 允许附加参数（如 --now），来自 config.AUTO_START.args
    需要管理员或有权限创建计划任务的用户。
    """
    if os.name != 'nt':
        return
    if not config.AUTO_START.get('enabled', True):
        LOGGER.info("AUTO_START 已禁用，不注册计划任务。")
        return

    task_name = config.AUTO_START.get('task_name', 'BTC_Trading_AutoStart')
    python_exe = os.sys.executable
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    main_py = os.path.join(project_dir, 'main.py')
    extra_args = config.AUTO_START.get('args', '')
    conda_env = config.AUTO_START.get('conda_env')

    # 优先使用Conda环境
    conda_exe = os.environ.get('CONDA_EXE')
    if conda_env and conda_exe and os.path.exists(conda_exe):
        LOGGER.info(f"检测到Conda环境，将使用 'conda run -n {conda_env}' 来构造自启动命令。")
        cmd_line = f'{_quote(conda_exe)} run -n {conda_env} python {_quote(main_py)} {extra_args}'.strip()
    else:
        # 降级为直接使用Python解释器
        cmd_line = f"{_quote(python_exe)} {_quote(main_py)} {extra_args}".strip()

    # schtasks /Create
    create_cmd = [
        'schtasks', '/Create', '/F',
        '/TN', task_name,
        '/TR', cmd_line,
        '/SC', 'ONLOGON',
        '/RL', 'HIGHEST'
    ]

    try:
        # 如果已存在，/F 会覆盖
        result = subprocess.run(create_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        if result.returncode == 0:
            LOGGER.success(f"已注册Windows自启动任务: {task_name}")
        else:
            LOGGER.warning(f"注册自启动任务失败: {result.stdout}")
    except Exception as e:
        LOGGER.error(f"注册自启动任务异常: {e}")


def remove_windows_autostart():
    """移除已注册的计划任务。"""
    if os.name != 'nt':
        return
    task_name = config.AUTO_START.get('task_name', 'BTC_Trading_AutoStart')
    try:
        result = subprocess.run(['schtasks', '/Delete', '/F', '/TN', task_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        if result.returncode == 0:
            LOGGER.success(f"已移除Windows自启动任务: {task_name}")
        else:
            LOGGER.warning(f"移除自启动任务失败: {result.stdout}")
    except Exception as e:
        LOGGER.error(f"移除自启动任务异常: {e}")



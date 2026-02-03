import os
import subprocess
import threading
import time
from typing import Optional

import config
from btc_predictor.utils import LOGGER


class AutoUpdater:
    """
    简单的Git自动更新器：
    - 周期性检查远端分支是否有新提交
    - 发现更新后执行 git fetch && git reset --hard origin/<branch>
    - 成功拉取后触发优雅重启（通过回调）
    """

    def __init__(self, repo_dir: Optional[str] = None, branch: Optional[str] = None,
                 interval_seconds: Optional[int] = None, on_updated=None):
        self.repo_dir = repo_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.branch = branch or config.AUTO_UPDATE.get('branch', 'main')
        self.interval_seconds = interval_seconds or config.AUTO_UPDATE.get('interval_seconds', 300)
        self.on_updated = on_updated
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not config.AUTO_UPDATE.get('enabled', False):
            LOGGER.info("AutoUpdater 已禁用。")
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name="AutoUpdater", daemon=True)
        self._thread.start()
        LOGGER.info(f"AutoUpdater 已启动，分支={self.branch}，间隔={self.interval_seconds}s")

    def stop(self):
        self._stop.set()

    def _run_cmd(self, cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=self.repo_dir, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def _has_remote_update(self) -> bool:
        # fetch
        r1 = self._run_cmd("git fetch --all")
        if r1.returncode != 0:
            LOGGER.warning(f"git fetch 失败: {r1.stdout}")
            return False
        # compare HEAD with origin/branch
        r2 = self._run_cmd(f"git rev-parse HEAD")
        r3 = self._run_cmd(f"git rev-parse origin/{self.branch}")
        if r2.returncode != 0 or r3.returncode != 0:
            LOGGER.warning(f"git rev-parse 失败: {r2.stdout} | {r3.stdout}")
            return False
        local = r2.stdout.strip()
        remote = r3.stdout.strip()
        return local != remote

    def _working_tree_clean(self) -> bool:
        r = self._run_cmd("git status --porcelain")
        return r.returncode == 0 and r.stdout.strip() == ""

    def _apply_update(self) -> bool:
        strategy = config.AUTO_UPDATE.get('update_strategy', 'hard_reset')
        protect = config.AUTO_UPDATE.get('protect_local_changes', True)
        if protect and not self._working_tree_clean():
            LOGGER.warning("检测到未提交的本地修改，已启用保护，跳过自动更新。")
            return False

        if strategy == 'pull_ff_only':
            cmd = f"git pull --ff-only origin {self.branch}"
        elif strategy == 'pull_rebase':
            cmd = f"git pull --rebase origin {self.branch}"
        elif strategy == 'pull_merge':
            cmd = f"git pull origin {self.branch}"
        else:  # hard_reset
            cmd = f"git reset --hard origin/{self.branch}"

        r = self._run_cmd(cmd)
        if r.returncode == 0:
            LOGGER.success(f"代码更新成功：{cmd}")
            return True
        LOGGER.error(f"自动更新命令失败: {cmd}\n{r.stdout}")
        return False

    def _run_loop(self):
        while not self._stop.is_set():
            try:
                if self._has_remote_update():
                    LOGGER.info("检测到远端更新，开始拉取...")
                    if self._apply_update():
                        if callable(self.on_updated):
                            try:
                                self.on_updated()
                            except Exception as e:
                                LOGGER.error(f"on_updated 回调执行失败: {e}")
                time.sleep(self.interval_seconds)
            except Exception as e:
                LOGGER.warning(f"AutoUpdater 循环异常: {e}")
                time.sleep(self.interval_seconds)


def graceful_restart():
    """通过重新启动当前Python进程来实现优雅重启。"""
    LOGGER.info("准备优雅重启进程...")
    python = os.sys.executable
    args = [python] + os.sys.argv
    os.execv(python, args)



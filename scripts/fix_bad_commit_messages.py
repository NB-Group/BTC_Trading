#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复重复的 commit messages，根据实际文件变更生成合理的提交信息。

使用方法:
    python scripts/fix_bad_commit_messages.py [--dry-run] [--since <date>]

选项:
    --dry-run: 只显示将要修改的内容，不实际执行
    --since: 只处理指定日期之后的 commits（格式: YYYY-MM-DD）
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Optional, Tuple

# 修复 Windows 控制台编码问题
if sys.platform == "win32":
    # 设置标准输出编码为 UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    # 设置环境变量
    os.environ["PYTHONIOENCODING"] = "utf-8"


def safe_print(*args, **kwargs):
    """安全地打印 Unicode 字符串，避免编码错误"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # 如果编码失败，使用 ASCII 安全的方式
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_args.append(arg.encode("ascii", errors="replace").decode("ascii"))
            else:
                safe_args.append(arg)
        print(*safe_args, **kwargs)

# 匹配有问题的 commit message 模式
BAD_PATTERNS = [
    r"^chore:\s*chore:\s*更新\s*\|\s*EN:\s*update\s*\|\s*EN:\s*update",
    r"^chore:\s*chore:\s*更新\s*\|\s*EN:\s*update",
    r"^chore:\s*更新\s*\|\s*EN:\s*update\s*\|\s*EN:\s*update",
    r"^chore:\s*chore:\s*chore:",
]


def run_git(*args: str) -> str:
    """运行 git 命令并返回输出"""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        safe_print(f"警告: git {' '.join(args)} 失败: {result.stderr}", file=sys.stderr)
    return result.stdout.strip()


def get_all_commits(since: Optional[str] = None) -> list[Tuple[str, str]]:
    """获取所有 commits，返回 (hash, message) 列表"""
    cmd = ["log", "--format=%H|%s", "--all"]
    if since:
        cmd.extend(["--since", since])
    output = run_git(*cmd)
    commits = []
    for line in output.split("\n"):
        if "|" in line:
            hash_val, message = line.split("|", 1)
            commits.append((hash_val.strip(), message.strip()))
    return commits


def is_bad_message(message: str) -> bool:
    """检查 commit message 是否有问题"""
    for pattern in BAD_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            return True
    return False


def analyze_file_changes(commit_hash: str) -> dict:
    """分析 commit 的文件变更，返回变更信息"""
    # 获取文件变更统计
    stat_output = run_git("show", "--stat", "--format=", commit_hash)
    
    # 获取实际的文件列表
    files_output = run_git("diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash)
    files = [f.strip() for f in files_output.split("\n") if f.strip()]
    
    # 分析变更类型
    added = 0
    deleted = 0
    modified = 0
    
    for line in stat_output.split("\n"):
        if "file changed" in line:
            # 解析类似 "2 files changed, 5 insertions(+), 3 deletions(-)"
            match = re.search(r"(\d+)\s+files?\s+changed", line)
            if match:
                modified = int(match.group(1))
            match = re.search(r"(\d+)\s+insertions?", line)
            if match:
                added = int(match.group(1))
            match = re.search(r"(\d+)\s+deletions?", line)
            if match:
                deleted = int(match.group(1))
    
    # 根据文件路径推断 scope 和 type
    scope = None
    change_type = "chore"
    
    # 分析文件路径来确定 scope
    scopes = defaultdict(int)
    for file in files:
        if "/" in file:
            parts = file.split("/")
            if parts[0] in ["execution_engine", "decision_engine", "scripts", "config", "githooks"]:
                scopes[parts[0]] += 1
    
    if scopes:
        scope = max(scopes.items(), key=lambda x: x[1])[0]
        # 移除下划线和复数形式
        scope = scope.replace("_engine", "").replace("_", "")
        if scope.endswith("s"):
            scope = scope[:-1]
    
    # 根据文件类型和变更推断 change_type
    has_py = any(f.endswith(".py") for f in files)
    has_config = any("config" in f.lower() or f.endswith((".toml", ".ini", ".yaml", ".yml")) for f in files)
    has_docs = any(f.endswith((".md", ".txt", ".rst")) for f in files)
    has_test = any("test" in f.lower() for f in files)
    
    if has_test:
        change_type = "test"
    elif has_docs:
        change_type = "docs"
    elif has_config:
        change_type = "chore"
    elif has_py:
        # 根据变更量判断是 feat 还是 fix
        if added > deleted * 2:
            change_type = "feat"
        elif deleted > added * 2:
            change_type = "refactor"
        else:
            change_type = "fix"
    
    return {
        "files": files,
        "added": added,
        "deleted": deleted,
        "modified": modified,
        "scope": scope,
        "type": change_type,
    }


def generate_commit_message(commit_hash: str, analysis: dict) -> str:
    """根据分析结果生成 commit message"""
    files = analysis["files"]
    scope = analysis["scope"]
    change_type = analysis["type"]
    
    # 生成简短描述
    if len(files) == 1:
        file_name = files[0].split("/")[-1].replace(".py", "").replace("_", " ")
        if scope:
            subject = f"更新 {scope} 模块的 {file_name}"
            en_subject = f"update {file_name} in {scope} module"
        else:
            subject = f"更新 {file_name}"
            en_subject = f"update {file_name}"
    elif scope:
        subject = f"更新 {scope} 相关代码"
        en_subject = f"update {scope} related code"
    else:
        subject = "更新代码"
        en_subject = "update code"
    
    # 根据 change_type 调整描述
    if change_type == "feat":
        subject = subject.replace("更新", "新增功能")
        en_subject = en_subject.replace("update", "add feature")
    elif change_type == "fix":
        subject = subject.replace("更新", "修复问题")
        en_subject = en_subject.replace("update", "fix issue")
    elif change_type == "refactor":
        subject = subject.replace("更新", "重构代码")
        en_subject = en_subject.replace("update", "refactor code")
    elif change_type == "docs":
        subject = subject.replace("更新", "更新文档")
        en_subject = en_subject.replace("update", "update docs")
    
    # 格式化 commit message
    if scope:
        return f"{change_type}({scope}): {subject} | EN: {en_subject}"
    else:
        return f"{change_type}: {subject} | EN: {en_subject}"


def main() -> None:
    parser = argparse.ArgumentParser(description="批量修复重复的 commit messages")
    parser.add_argument("--dry-run", action="store_true", help="只显示将要修改的内容，不实际执行")
    parser.add_argument("--since", type=str, help="只处理指定日期之后的 commits (格式: YYYY-MM-DD)")
    parser.add_argument("--yes", "-y", action="store_true", help="自动确认，不询问用户")
    args = parser.parse_args()
    
    safe_print("正在查找有问题的 commit messages...")
    
    commits = get_all_commits(args.since)
    bad_commits = [(h, m) for h, m in commits if is_bad_message(m)]
    
    if not bad_commits:
        safe_print("未找到需要修复的 commits")
        return
    
    safe_print(f"\n找到 {len(bad_commits)} 个需要修复的 commits:\n")
    
    # 分析每个 commit 并生成新的 message
    fixes = []
    for commit_hash, old_message in bad_commits:
        safe_print(f"分析 {commit_hash[:8]}...")
        analysis = analyze_file_changes(commit_hash)
        new_message = generate_commit_message(commit_hash, analysis)
        
        fixes.append((commit_hash, old_message, new_message))
        
        # 安全地打印消息，避免编码问题
        old_preview = old_message[:60] if len(old_message) > 60 else old_message
        safe_print(f"  旧: {old_preview}")
        safe_print(f"  新: {new_message}")
        safe_print()
    
    if args.dry_run:
        safe_print("\n[DRY RUN] 以上是预览，未实际修改")
        return
    
    # 生成 rebase script
    safe_print("\n生成 rebase 脚本...")
    script_lines = []
    for commit_hash, old_message, new_message in reversed(fixes):
        # 使用 reword 来修改 commit message
        script_lines.append(f"reword {commit_hash} {new_message}")
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for line in script_lines:
            f.write(line + "\n")
        script_file = f.name
    
    safe_print(f"\nRebase 脚本已保存到: {script_file}")
    safe_print("\n要执行修复，请运行:")
    safe_print(f"  git rebase -i {fixes[-1][0]}^")
    safe_print("\n或者使用以下命令:")
    safe_print(f"  GIT_SEQUENCE_EDITOR='python scripts/fix_bad_commit_messages.py --apply' git rebase -i {fixes[-1][0]}^")
    
    # 如果用户想要自动执行，可以提供一个选项
    if args.yes:
        response = "y"
    else:
        safe_print("\n是否现在执行 rebase? (y/N): ", end="")
        try:
            response = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            safe_print("\n已取消。你可以稍后手动执行修复。")
            return
    
    if response == "y":
        # 使用 git filter-branch 或 git rebase 来批量修改
        safe_print("\n使用 git filter-branch 进行批量修改...")
        # 创建一个临时的 rewrite 脚本
        rewrite_script = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
        rewrite_script.write("#!/usr/bin/env python3\n")
        rewrite_script.write("import os, sys\n")
        rewrite_script.write("commit = os.environ.get('GIT_COMMIT')\n")
        rewrite_script.write("mapping = {\n")
        for commit_hash, old_message, new_message in fixes:
            # 转义引号
            new_msg_escaped = new_message.replace('"', '\\"').replace('\n', '\\n')
            rewrite_script.write(f'    "{commit_hash}": "{new_msg_escaped}\\n",\n')
        rewrite_script.write("}\n")
        rewrite_script.write("if commit in mapping:\n")
        rewrite_script.write("    sys.stdout.buffer.write(mapping[commit].encode('utf-8'))\n")
        rewrite_script.write("else:\n")
        rewrite_script.write("    sys.stdout.buffer.write(sys.stdin.buffer.read())\n")
        rewrite_script.close()
        
        os.chmod(rewrite_script.name, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        # 找到最早的 commit
        oldest_hash = fixes[-1][0]
        
        safe_print(f"\n执行 git filter-branch (从 {oldest_hash[:8]} 开始)...")
        safe_print("这可能需要一些时间...")
        
        # 在 Windows 上，路径需要正确引用和转义
        # 获取绝对路径并规范化
        abs_script_path = os.path.abspath(rewrite_script.name)
        # 在 Windows 上，使用正斜杠或正确转义的反斜杠
        if sys.platform == "win32":
            # Windows 上，路径中的反斜杠需要转义，或者使用正斜杠
            # 使用正斜杠通常更安全
            normalized_path = abs_script_path.replace("\\", "/")
            # 使用双引号包裹路径，确保空格和特殊字符被正确处理
            msg_filter_cmd = f'python "{normalized_path}"'
        else:
            msg_filter_cmd = f"python {shlex.quote(abs_script_path)}"
        
        result = subprocess.run(
            [
                "git",
                "filter-branch",
                "--msg-filter",
                msg_filter_cmd,
                f"{oldest_hash}^..HEAD",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        
        if result.returncode == 0:
            safe_print("\n✅ 修复完成!")
            safe_print("\n请检查结果:")
            safe_print("  git log --oneline -20")
            safe_print("\n如果满意，可以运行:")
            safe_print("  git push --force-with-lease")
        else:
            safe_print(f"\n❌ 修复失败: {result.stderr}")
            sys.exit(1)
    else:
        safe_print("\n已取消。你可以稍后手动执行修复。")


if __name__ == "__main__":
    main()


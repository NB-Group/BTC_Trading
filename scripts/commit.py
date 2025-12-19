#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Git 提交脚本，用于提交代码更改，避免编码问题
支持命令行参数或交互式输入提交信息
"""
import subprocess
import sys
import os
import tempfile
import argparse
import io

# 设置标准输出编码为 UTF-8（Windows 兼容）
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass

def run_command(cmd, description, cwd=None):
    """执行命令并处理编码"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*60}")
    
    try:
        # 使用 subprocess 执行命令，设置编码为 UTF-8
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'  # 遇到编码错误时替换而不是失败
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode != 0:
            print(f"[错误] 命令执行失败，返回码: {result.returncode}")
            return False
        else:
            print(f"[成功] 命令执行成功")
            return True
    except Exception as e:
        print(f"[错误] 执行命令时出错: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Git 提交脚本，避免编码问题')
    parser.add_argument('-m', '--message', type=str, help='提交信息（单行）')
    parser.add_argument('-f', '--file', type=str, help='从文件读取提交信息')
    parser.add_argument('--add-all', action='store_true', help='自动添加所有更改的文件')
    parser.add_argument('--no-push', action='store_true', help='不推送到远程（默认会推送）')
    
    args = parser.parse_args()
    
    # 获取仓库根目录
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("="*60)
    print("Git 提交脚本")
    print(f"仓库路径: {repo_root}")
    print("="*60)
    
    # 1. 检查 git 状态
    if not run_command("git status", "检查 Git 状态", cwd=repo_root):
        print("[错误] Git 状态检查失败，请确保在 Git 仓库中")
        return 1
    
    # 2. 添加修改的文件
    if args.add_all:
        print("\n[信息] 添加所有更改的文件...")
        if not run_command("git add -A", "添加所有更改", cwd=repo_root):
            print("[警告] 添加文件失败")
            return 1
    else:
        print("\n[信息] 检查是否有未暂存的更改...")
        result = subprocess.run(
            "git diff --name-only",
            shell=True,
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.stdout.strip():
            print("[提示] 发现未暂存的更改，请先使用 'git add' 添加文件")
            print("或者使用 --add-all 参数自动添加所有更改")
            return 1
    
    # 3. 检查暂存区状态
    print("\n[信息] 检查暂存区状态...")
    run_command("git status", "查看暂存区状态", cwd=repo_root)
    
    # 4. 获取提交信息
    commit_message = None
    
    if args.file:
        # 从文件读取
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                commit_message = f.read().strip()
        except Exception as e:
            print(f"[错误] 无法读取文件 {args.file}: {e}")
            return 1
    elif args.message:
        # 从命令行参数获取
        commit_message = args.message
    else:
        # 交互式输入
        print("\n[提示] 请输入提交信息（多行，以空行结束或按 Ctrl+Z 然后 Enter 结束）:")
        print("="*60)
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        commit_message = '\n'.join(lines).strip()
    
    if not commit_message:
        print("[错误] 提交信息不能为空")
        return 1
    
    # 5. 提交更改
    print("\n[信息] 提交更改...")
    try:
        print(f"提交信息:\n{commit_message}\n")
    except UnicodeEncodeError:
        print("[信息] 提交信息（包含中文）已准备就绪\n")
    
    # 将提交信息写入临时文件，避免命令行编码问题
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False, newline='\n') as f:
        f.write(commit_message)
        temp_file = f.name
    
    try:
        # 使用 -F 参数从文件读取提交信息，避免编码问题
        success = run_command(
            f'git commit -F "{temp_file}"',
            "提交更改",
            cwd=repo_root
        )
        
        if success:
            print("\n" + "="*60)
            print("[成功] 提交成功！")
            print("="*60)
            
            # 6. 推送到远程（如果需要）
            if not args.no_push:
                print("\n[信息] 推送到远程仓库...")
                run_command("git push", "推送到远程", cwd=repo_root)
            
            return 0
        else:
            print("\n" + "="*60)
            print("[错误] 提交失败，请检查错误信息")
            print("="*60)
            return 1
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_file)
        except:
            pass

if __name__ == "__main__":
    sys.exit(main())


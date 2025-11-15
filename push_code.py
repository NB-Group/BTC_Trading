#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git 推送脚本，防止乱码问题
自动添加、提交并推送更改
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """执行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print('='*60)
    
    # 设置环境变量防止乱码
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LANG'] = 'zh_CN.UTF-8'
    env['LC_ALL'] = 'zh_CN.UTF-8'
    
    # 在 Windows 上设置代码页为 UTF-8
    if sys.platform == 'win32':
        try:
            subprocess.run(['chcp', '65001'], shell=True, check=False, capture_output=True)
        except:
            pass
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return False, "", str(e)

def main():
    """主函数"""
    print("="*60)
    print("Git 推送脚本 - 防止乱码版本")
    print("="*60)
    
    # 1. 检查状态
    success, stdout, stderr = run_command('git status', '检查 Git 状态')
    if not success:
        print("\n❌ Git 状态检查失败")
        return False
    
    # 2. 检查是否有未提交的更改
    result = subprocess.run(
        'git diff --quiet',
        shell=True,
        capture_output=True
    )
    has_unstaged = result.returncode != 0
    
    result = subprocess.run(
        'git diff --cached --quiet',
        shell=True,
        capture_output=True
    )
    has_staged = result.returncode != 0
    
    if not has_unstaged and not has_staged:
        print("\n✅ 没有待提交的更改，直接推送...")
    else:
        # 3. 添加所有更改
        if has_unstaged:
            print("\n📦 添加所有更改...")
            success, _, _ = run_command('git add -A', '添加所有更改')
            if not success:
                print("\n❌ 添加文件失败")
                return False
        
        # 4. 提交更改（使用 commitizen 格式）
        commit_msg = "fix(decision_engine): 添加余额为0时的平仓操作说明"
        print(f"\n💾 提交更改: {commit_msg}")
        success, _, _ = run_command(f'git commit -m "{commit_msg}"', '提交更改')
        if not success:
            print("\n❌ 提交失败")
            return False
    
    # 5. 推送到远程
    print("\n🚀 推送到远程仓库...")
    success, stdout, stderr = run_command('git push', '推送到远程仓库')
    if not success:
        print("\n❌ 推送失败")
        print("错误信息:", stderr)
        return False
    
    print("\n✅ 推送成功!")
    print("="*60)
    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}", file=sys.stderr)
        sys.exit(1)

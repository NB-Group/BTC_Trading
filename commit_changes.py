#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Git 提交脚本，用于提交代码更改，避免编码问题
"""
import subprocess
import sys
import os

def run_command(cmd, description):
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
    print("="*60)
    print("Git 提交脚本")
    print("="*60)
    
    # 1. 检查 git 状态
    if not run_command("git status", "检查 Git 状态"):
        print("[错误] Git 状态检查失败，请确保在 Git 仓库中")
        return
    
    # 2. 添加所有修改的文件
    print("\n[信息] 添加修改的文件...")
    if not run_command("git add -A", "添加所有更改"):
        print("[警告] 添加文件失败")
        return
    
    # 3. 检查暂存区状态
    print("\n[信息] 检查暂存区状态...")
    run_command("git status", "查看暂存区状态")
    
    # 4. 提交更改
    commit_message = """fix(execution): 修复可用保证金与账户总资产概念混淆 | EN: fix confusion between available margin and total equity

- 明确区分"可用保证金"（availEq）和"账户总资产"（equity）的概念
- 更新 get_balance() 方法文档，明确返回的是可用保证金而非账户总资产
- 添加账户总资产的获取和日志记录，便于调试和问题排查
- 当可用保证金为0但账户总资产不为0时，记录警告说明资金可能被用于持仓
- 更新决策引擎提示，将"当前账户余额"改为"当前可用保证金"
- 在杠杆计算公式中明确使用可用保证金，并添加概念澄清说明
- 改进错误信息，明确说明可用保证金为0不等于账户总资产为0"""
    
    print("\n[信息] 提交更改...")
    # 将提交信息写入临时文件，避免命令行编码问题
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False, newline='\n') as f:
        # 确保使用 UTF-8 编码写入
        f.write(commit_message)
        temp_file = f.name
    
    try:
        # 使用 -F 参数从文件读取提交信息，避免编码问题
        success = run_command(
            f'git commit -F "{temp_file}"',
            "提交更改"
        )
        
        if success:
            print("\n" + "="*60)
            print("[成功] 提交成功！")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("[错误] 提交失败，请检查错误信息")
            print("="*60)
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_file)
        except:
            pass

if __name__ == "__main__":
    main()

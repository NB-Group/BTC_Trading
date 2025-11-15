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
    
    # 2. 添加修改的文件
    print("\n[信息] 添加修改的文件...")
    files_to_add = [
        "execution_engine/okx_trader.py",
        "commit_changes.py"
    ]
    
    for file in files_to_add:
        if os.path.exists(file):
            if not run_command(f'git add "{file}"', f"添加文件: {file}"):
                print(f"[警告] 添加文件 {file} 失败")
        else:
            print(f"[警告] 文件不存在，跳过: {file}")
    
    # 5. 检查暂存区状态
    print("\n[信息] 检查暂存区状态...")
    run_command("git status", "查看暂存区状态")
    
    # 6. 提交更改
    commit_message = """feat: 增强交易订单错误处理和日志记录

- 为所有 create_order 调用添加完整的异常捕获（ExchangeError、NetworkError）
- 改进错误信息格式，包含异常类型、消息和完整响应
- 添加详细的调试日志，记录下单参数和返回值
- 覆盖所有下单场景：开仓、平仓、止损单、止盈单
- 使用异常链（from e）保留原始异常信息，便于问题排查"""
    
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


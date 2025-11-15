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
            print(f"❌ 命令执行失败，返回码: {result.returncode}")
            return False
        else:
            print(f"✅ 命令执行成功")
            return True
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("Git 提交脚本")
    print("="*60)
    
    # 1. 检查 git 状态
    if not run_command("git status", "检查 Git 状态"):
        print("❌ Git 状态检查失败，请确保在 Git 仓库中")
        return
    
    # 2. 添加修改的文件
    print("\n📝 添加修改的文件...")
    files_to_add = [
        "execution_engine/okx_trader.py",
        "scripts/fix_commit_messages.py"
    ]
    
    for file in files_to_add:
        if os.path.exists(file):
            if not run_command(f'git add "{file}"', f"添加文件: {file}"):
                print(f"⚠️  警告: 添加文件 {file} 失败")
        else:
            print(f"⚠️  文件不存在，跳过: {file}")
    
    # 3. 添加删除的文件
    print("\n🗑️  处理删除的文件...")
    deleted_files = [
        "messages.txt",
        "messages2.txt",
        "requirements-dev.txt"
    ]
    
    for file in deleted_files:
        if run_command(f'git rm "{file}"', f"删除文件: {file}"):
            print(f"✅ 已标记删除: {file}")
        else:
            print(f"⚠️  文件可能已不存在: {file}")
    
    # 4. 添加新文件（如果有）
    new_files = [
        "scripts/fix_bad_commit_messages.py"
    ]
    
    for file in new_files:
        if os.path.exists(file):
            if not run_command(f'git add "{file}"', f"添加新文件: {file}"):
                print(f"⚠️  警告: 添加新文件 {file} 失败")
    
    # 5. 检查暂存区状态
    print("\n📋 检查暂存区状态...")
    run_command("git status", "查看暂存区状态")
    
    # 6. 提交更改
    commit_message = """fix: 修复止损单参数和优化仓位计算逻辑

- 修复止损单参数：使用 CCXT 标准参数 triggerPrice 和 orderType
- 优化仓位计算：添加详细的日志输出，改进 suggested_trade_size 的处理逻辑
- 支持 suggested_trade_size > 1 的情况（视为张数）
- 添加仓位计算的详细日志，便于调试和排查问题"""
    
    print("\n💾 提交更改...")
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
            print("✅ 提交成功！")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ 提交失败，请检查错误信息")
            print("="*60)
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_file)
        except:
            pass

if __name__ == "__main__":
    main()


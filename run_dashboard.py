#!/usr/bin/env python3
"""
启动BTC智能决策看板的便捷脚本
"""

import subprocess
import sys
import os

def main():
    """启动Streamlit dashboard"""
    print("🚀 正在启动BTC智能决策看板...")
    print("📊 看板将在浏览器中自动打开")
    print("💡 按 Ctrl+C 可停止看板服务")
    print("-" * 50)
    
    try:
        # 确保在正确的目录中运行
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # 启动Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.serverAddress", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 看板服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动看板失败: {e}")
        return 1
    except FileNotFoundError:
        print("❌ 未找到streamlit，请先安装: pip install streamlit")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
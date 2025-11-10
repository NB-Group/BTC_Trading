#!/usr/bin/env python3
"""Rewrite specific commit messages with proper UTF-8 content.

This script is intended to be used with `git filter-branch --msg-filter`
to fix garbled Chinese commit subjects that were produced when commit
messages were incorrectly encoded on Windows. It maps the rewritten
commits by SHA and emits the desired UTF-8 message while leaving other
commits untouched.
"""

from __future__ import annotations

import os
import sys


MAPPING = {
    "0eca3d2172a4c08a7505ee629009d6e456658087": (
        "chore(scripts): 规范 rewrite_cb 首行解析并去重 | EN: normalize rewrite_cb header parsing\n"
    ),
    "5e9a1b472da93c920cb2252520d2be27384e9434": (
        "chore(githooks): 新增 rewrite_cb 并确保兼容无 commitizen 环境 | EN: add rewrite helper and guard commit-msg without commitizen\n"
    ),
    "aca36b80ecf263ecd5a7bd857d0460b133f7fc07": (
        "chore: 引入提交模板与 commitizen 钩子 | EN: add commit template and commitizen hooks\n"
    ),
    "8515a3d3737b63a3d5163e9d9eb6096b43ac8329": (
        "feat(report): 注入持仓盈亏快照并邮件展示 | EN: add position snapshot to reports\n"
    ),
    "3e7cd915dc324ef0a74b7f9a03ccf2d30cae21fe": (
        "docs(ai): 强化统一分析提示词的趋势策略要求 | EN: clarify unified analyzer trading guidance\n"
    ),
    "e25abec66fbd039167799ac0020afd607a91901d": (
        "feat(engine): 优化统一 Gemini 回退并强化 OKX 交易守护 | EN: improve unified Gemini fallback and OKX safeguards\n"
    ),
    "32d8c401bb84083840cc0d7b9f0ecb1e98765fdb": (
        "feat(decision): 接入 Gemini 多模态决策路径 | EN: add unified Gemini decision flow\n"
    ),
    "085d02847cfddb7712c94a2af0d3b63268473449": (
        "feat(vlm): 加强流式重试并包装思考输出 | EN: harden VLM streaming retries\n"
    ),
    "4b8fba90a4e0b78a470b4f5917e050d3e01aa622": (
        "chore(vlm): 拆分连接与流式超时参数 | EN: split VLM connect and stream timeouts\n"
    ),
    "0b01c55acb876bc7978727367f276671c817aaa4": (
        "feat(vlm): 启用 SSE 流式输出并提升生成上限 | EN: enable VLM streaming with higher token limit\n"
    ),
    "afcb8da8b2ffb54b7b33a853606b86298ca2af5b": (
        "chore: 更新 | EN: update\n"
    ),
    "39ce90c45f0dee73e32e9cfbc7106cbe093758d0": (
        "chore: 更新 | EN: update\n"
    ),
    "e12a274f1d32af3d8a08e4f4e4d674351aa75c88": (
        "chore: 更新 | EN: update\n"
    ),
    "63af408a0ac11cc564473d4b47178d48fd36ddfa": (
        "chore: 更新 | EN: update\n"
    ),
    "4abcf8f9f2e9631112101a4f552f70b476a74478": (
        "chore: 更新 | EN: update\n"
    ),
    "1c5088d24d13db4bdc251d6d901b4390bfd2c19d": (
        "chore: 更新 | EN: update\n"
    ),
    "149a770c965aa5f4c94b4b299f40af3ebcd99b42": (
        "chore: 更新 | EN: update\n"
    ),
    "7efd060b6b36e7e823597ce3fea384dc00949350": (
        "chore: 更新 | EN: update\n"
    ),
    "aea188d7c7957776c109525f494bca9e79744657": (
        "chore: 更新 | EN: update\n"
    ),
    "ef6a9f5446df852b6fe55af65b7eb8df23b0cb13": (
        "chore: 更新 | EN: update\n"
    ),
    "1333090e806a08aa8275643a7f5d065bd348947e": (
        "chore: 更新 | EN: update\n"
    ),
    "c12bb0089ab2461ca5eb8fd22f39264c4c1cddba": (
        "chore: 更新 | EN: update\n"
    ),
    "2417eaf9f0a95ead39a0614d3c4de0d43cac8718": (
        "chore: 更新 | EN: update\n"
    ),
    "f439de838e1ec6a070033db1bb1d1180f12b2887": (
        "chore: v1.0.13 又更新了一下 | EN: update\n"
    ),
    "e5df9e238476bc8a3b8eb698f6ed8288b543460d": (
        "chore: Merge branch 'main' of origin | EN: update\n"
    ),
    "e5da2352cd7d0ea8e72842f29c2f174446aba668": (
        "chore: Update README.md | EN: update\n"
    ),
    "d2deaa804a283f94480c2dae09fa53c78daddf69": (
        "chore: v1.0.12 更新了一下 | EN: update\n"
    ),
    "361c54780613da9f8435b88cd345fd873fb5decc": (
        "chore: v1.0.11 升级部分功能 | EN: update\n"
    ),
    "1b4edb7df5fd81cc4177c83e26ee6fc3639a5976": (
        "chore: v1.0.10 修复一直空仓的问题 | EN: update\n"
    ),
    "ab16b84d7cf97ca5d9ed7c940e5998373f2aae0a": (
        "chore: v0.0.9 修复交易逻辑bug | EN: update\n"
    ),
    "434aef56dcd4532350cfc758a8df4af9b36bbb06": (
        "chore: v1.0.9 修复平仓失败问题 | EN: update\n"
    ),
    "5523ca0046950df3c91d29310670b1d28448ea00": (
        "chore: v1.0.9 修复平仓失败问题 | EN: update\n"
    ),
    "37912dd47c06df4ae336b2cfc975d9394768de63": (
        "chore: v1.0.8 扩充一点点的邮件内容 | EN: update\n"
    ),
    "cf58ca6c6970ee3eefe2ef0f1261b844806aaa2f": (
        "chore: v1.0.7 将CoinDesk RSS获取方式改为Playright获取 | EN: update\n"
    ),
    "02b27ba02dd30dd2512a6f138075bcbd0e89a337": (
        "chore: v1.0.7 将CoinDesk RSS获取方式改为Playright获取 | EN: update\n"
    ),
    "eed20cc8fb7d277679582e2afd5d976290b95554": (
        "chore: v1.0.6 更新 | EN: minor tweaks\n"
    ),
    "de3fdfb2dadd47aabe2c7c7773a701e47a95a24f": (
        "chore: v1.0.5 更新 | EN: minor tweaks\n"
    ),
    "e44f95c9845af176abdc6066ccf0191c41f89c59": (
        "chore: v1.0.4 更新 | EN: minor tweaks\n"
    ),
    "451a1cd947ec5481bec56034d944660f918aa694": (
        "chore: v1.0.3 更新了一些东西 | EN: update\n"
    ),
    "5dc23bb474a5e9b06fb097e9799cd43fa45a3556": (
        "chore: v1.0.2 修复一些问题 | EN: update\n"
    ),
    "5bbbf282d6368b46c399489e088cf142048fcd2a": (
        "chore: Merge branch 'main' of https://github.com/NB-Group/BTC_Trading | EN: update\n"
    ),
    "e857c16eb1b0ef83aef217573c84df7449df45df": (
        "chore: Update LICENSE | EN: update\n"
    ),
    "507d9fb83cb6d390f42f5299d83895035eeed9a6": (
        "chore: v1.0.1 修复k线生成问题 | EN: update\n"
    ),
    "14591cab3ddc2acaf6aaaed267ffb710bd35a1cf": (
        "chore: v1.0.0 发布版 | EN: update\n"
    ),
}


def main() -> None:
    commit = os.environ.get("GIT_COMMIT")
    original = sys.stdin.buffer.read()

    if commit in MAPPING:
        sys.stdout.buffer.write(MAPPING[commit].encode("utf-8"))
    else:
        sys.stdout.buffer.write(original)


if __name__ == "__main__":
    main()


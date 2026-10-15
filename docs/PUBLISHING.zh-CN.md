# FADetr 发布说明

项目名称已确定为 **FADetr**。本目录是基于 O²-RT-DETR 的项目扩展，需要完整上游框架。
目前还没有完成独立环境中的训练与复现验证。开源有助于审稿人检查和复现方法，
不能代替创新性、实验质量和公平对比。

## 主入口与本地检查

主配置：`configs/o2_rtdetr_r34vd_2xb4_72e_dota_daff_faahead.py`。
当前代码使用 ResNet34-vd、DAFFusionFPN、HybridEncoder、4 层 AGA Decoder 和
RotatedRTDETRHead。保留历史配置名，没有因为改名为 FADetr 改动模型计算。

在本目录运行静态检查：

```bash
python tools/check_release.py
```

按 README 将 `configs/`、`rotated_rtdetr/`、`tools/` 放到兼容上游环境的
`projects/rotated_rtdetr/` 下，再从上游根目录运行：

```bash
python projects/rotated_rtdetr/tools/check_runtime.py
```

后者会构建、初始化模型，可能下载预训练权重；不验证前向输出、训练收敛或精度。
当前准备环境没有 torch/MMEngine/MMCV/MMDetection/MMRotate，未执行模型验证。

## 发布仍需补齐

1. 账号已连接为 `Chipssr`，计划仓库为 `Chipssr/FADetr`，远程仓库尚待创建。
2. 训练机器的实际环境导出，以及上游代码的准确提交 SHA。
3. 最佳权重、完整训练/评估日志、解析后的配置与真实指标。
4. 独立评估划分。当前训练和验证路径相同，不能直接作为独立性能证据。
5. 上游 LICENSE 已原样获取；已记录参考提交并标注不同的模型文件，训练源版本仍需确认。

实际配置每卡 batch size 为 8，文件名中的 `2xb4` 不代表最终批量。
权重、数据、缓存、日志已加入 `.gitignore`，本次未删除这些本地文件。
其他实验配置保留，部分依赖缺失的旧类或外部项目，不作为发布支持入口。

## 建立远程仓库后

确认计划发布的文件，再推送到你拥有的 GitHub 仓库。以下地址是占位符：

```bash
git add .
git diff --cached --stat
git commit -m "Prepare FADetr research code release"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/FADetr.git
git push -u origin main
```

本地首个源码提交将随此次整理建立，远程推送尚未执行。最终发布后，记录固定提交或 release tag，让论文结果对应同一版代码。
如目标期刊/会议要求匿名评审，按其当时规则处理仓库链接和作者信息。

# FADetr 发布与后续更新

FADetr 已发布到公开仓库：[Chipssr/FADetr](https://github.com/Chipssr/FADetr)。
本目录是基于 O²-RT-DETR 的项目扩展，需要完整兼容的上游框架。
目前尚未完成独立环境中的训练与精度复现验证；GitHub 自动检查只验证源码与配置结构。

## 已完成

- 上传源代码、配置、安装与训练说明、许可证和来源记录。
- 配置 `.gitignore`，排除数据、权重、缓存、日志及常见本地凭据文件。
- 核对首次上传的远程文件树与本地检查过的快照完全一致。
- 首次发布的 [GitHub 静态检查](https://github.com/Chipssr/FADetr/actions/runs/38718114843) 已通过。

## 主入口与检查

主配置：`configs/o2_rtdetr_r34vd_2xb4_72e_dota_daff_faahead.py`。
当前模型使用 ResNet34-vd、DAFFusionFPN、HybridEncoder、4 层 AGA Decoder 和
RotatedRTDETRHead。保留历史配置名，开源整理没有改变模型计算。

在 FADetr 仓库根目录运行静态检查：

```bash
python tools/check_release.py
```

按 README 将 `configs/`、`rotated_rtdetr/`、`tools/` 放到兼容上游环境的
`projects/rotated_rtdetr/` 下，再从上游根目录运行：

```bash
python projects/rotated_rtdetr/tools/check_runtime.py
```

后者会构建、初始化模型，可能下载预训练权重；不验证前向输出、训练收敛或精度。
发布准备环境没有 torch/MMEngine/MMCV/MMDetection/MMRotate，尚未执行模型验证。

## 论文复现仍需补齐

1. 训练机器的实际环境导出，以及上游训练代码的准确提交 SHA。
2. 最佳权重、完整训练/评估日志、解析后的配置与真实指标。
3. 独立评估划分。当前训练和验证路径相同，不能直接作为独立性能证据。
4. 上游 LICENSE 已原样获取；已记录参考提交并标注不同的模型文件，训练源版本仍需确认。

实际配置每卡 batch size 为 8，文件名中的 `2xb4` 不代表最终批量。
其他实验配置保留，部分依赖缺失的旧类或外部项目，不作为发布支持入口。

## 后续更新

从新的电脑获取代码：

```bash
git clone https://github.com/Chipssr/FADetr.git
cd FADetr
```

更新代码后，在已配置好本机 GitHub 身份验证的仓库中提交并推送：

```bash
python tools/check_release.py
git add .
git diff --cached --stat
git commit -m "Update FADetr"
git push origin main
```

论文实验完成后记录固定提交或 release tag，确保结果能对应到同一版代码。
如目标期刊/会议要求匿名评审，按其当时规则处理公开仓库链接和作者信息。

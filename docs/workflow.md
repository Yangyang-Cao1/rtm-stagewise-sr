# Datasets Crop Stagewise SR + RTM Workflow Updated

## 1. 文档目的

这份文档用于统一说明当前已经落地的完整流程，包括：

- 旧的 `project_B4/` 目录组织下的 `PRISMA + Sentinel-2 + field CSV`
- 新增的“数据直接放在数据集根目录”的 `EnMAP + Sentinel-2 + field CSV`
- `dataset_onepatch_centered_plot15_areaweighted_v2` 的最新生成规则
- 当前稳定使用的 `stagewise + RTM teacher + alpha fusion + post smoothing` 主线
- 反射率量纲修正、可视化修正、过期结果自动判定等更新


## 2. 当前主线方法一句话总结

当前稳定方案仍然是：

1. 先做 `Stage 1 no-RTM`，拿到空间结构稳定的 HR-HSI
2. 再做一个偏保守的 `veg_lowfreq` baseline
3. 再用 `RTM prior` 训练一个 teacher
4. 最后在 `Stage 2` 中不直接自由修改整幅图，而是学习按波长变化的 `alpha(lambda)`，在 `baseline` 和 `teacher` 之间做 band-wise 融合
5. 最后做轻量的 post spectral smoothing

核心原则没有变：

- 空间细节主要继承 `Stage 1`
- 植被关键谱段更多借助 `teacher`
- RTM 是物理先验和教师信号，不直接替代最终图像


## 3. 和旧文档相比的主要更新

这次更新最重要的变化有 6 点：

1. 现在同时支持两类输入组织：
   - `datasets_crop/<dataset>/project_B4/`
   - `datasets_crop/<dataset>/` 根目录直放输入文件
2. 新增了专门面向根目录 `EnMAP single-date` 数据的批处理入口
3. `HSI` 输入现在会显式做反射率量纲检查和缩放，避免把 `EnMAP int16` 直接当作 `0~1` 反射率
4. 可视化现在会忽略 `EnMAP` 的 `nodata=-32768`，不会再把 HSI 拉伸成接近黑白图
5. onepatch 和 run 现在都有“过期判定”，旧版本数据和旧结果不会被静默复用
6. 诊断图和主可视化的 HSI RGB 波段选择已经统一为按波长选 `664.5 / 560.0 / 496.6 nm`


## 4. 目录与环境

### 4.1 主要目录

- 项目主目录：`../rtm_stagewise_sr`
- 数据目录：`../datasets_crop`
- onepatch 生成脚本目录：`../rtm_stagewise_sr/preprocessing`
- RTM 目录：`../rtm_stagewise_sr/rtm`

### 4.2 推荐运行环境

推荐使用仓库提供的 Conda 环境：

```bash
conda env create -f environment.yml
conda activate rtm-stagewise-sr
```

说明：

- 推荐 Python 3.10
- 环境需要包含 `rasterio`、`pyproj`、PyTorch 等依赖


## 5. 当前关键脚本

### 5.1 主入口

- PRISMA 与 EnMAP 统一批处理入口：[batch.py](../rtm_stagewise_sr/batch.py)
- 单数据集完整流程入口：[pipeline.py](../rtm_stagewise_sr/pipeline.py)

### 5.2 数据生成与几何

- CSV -> centered onepatch：[prepare.py](../rtm_stagewise_sr/preprocessing/prepare.py)

- centered + area-weighted 几何与归一化工具：[geometry.py](../rtm_stagewise_sr/preprocessing/geometry.py)

### 5.3 训练与评估

- Stagewise 训练：[training.py](../rtm_stagewise_sr/training.py)

- RTM-only teacher：[teacher.py](../rtm_stagewise_sr/rtm/teacher.py)

- field 评估：[evaluation.py](../rtm_stagewise_sr/evaluation.py)

- 40 点光谱图：[plots.py](../rtm_stagewise_sr/plots.py)

- 可视化：[visualization.py](../rtm_stagewise_sr/visualization.py)


## 6. 目前支持的两类输入组织

### 6.1 旧格式：`project_B4/`

每个数据集结构为：

```text
datasets_crop/<dataset_name>/project_B4/
├── prisma_filtered_bands_*.tif
├── S2_L2A_Median_CloudFree_*.tif
├── filtered_wavelengths_*.npy
└── *.csv
```

由统一的 [batch.py](../rtm_stagewise_sr/batch.py) 自动扫描处理。

### 6.2 新格式：根目录直放的 `EnMAP single-date`

每个数据集结构为：

```text
datasets_crop/<dataset_name>/
├── enmap_filtered_bands_*.tif
├── S2_single_date_*.tif
├── enmap_filtered_wavelengths_*.npy
└── *.csv
```

同样由统一的 [batch.py](../rtm_stagewise_sr/batch.py) 自动扫描；也可在命令后显式给出数据集目录名。


## 7. field CSV 的当前假设

[prepare.py](../rtm_stagewise_sr/preprocessing/prepare.py) 和评估流程默认仍然使用下面这些约定：

- 第 0 行包含 field 光谱波长
- 第 6 列是纬度
- 第 7 列是经度
- 第 8 列是 plot id
- 能从第 0 行解析成数值的列会被视为光谱列
- 一行至少有 2 个有效光谱值才算有效 field 点

补充说明：

- 如果 plot id 重复，脚本会自动补成唯一 id
- 如果有效点不足 40 个，流程仍然可以正常运行


## 8. onepatch 数据生成的当前标准

### 8.1 方法名

当前统一采用：

- `centered_plot15_areaweighted`

产出目录名是：

- `dataset_onepatch_centered_plot15_areaweighted_v2`

### 8.2 几何核心

对每个 field plot：

1. 从 CSV 读取中心经纬度
2. 把 plot 视为 `15m x 15m` 的面，而不是单点
3. 投影到 HSI / S2 的 CRS
4. 计算它与 HSI、S2 像元的实际面积重叠
5. 保存 footprint 像元和面积权重

对整批 plot：

1. 先求所有 plot 的包围范围
2. 在 HSI 网格上生成 centered patch
3. patch 尺寸按 `ALIGN=128` 对齐
4. 再把相同地理范围映射到 S2 上切出对应 patch

### 8.3 关键函数

- `window_to_cover_bounds_centered(...)`
- `footprint_pixels(...)`
- `bbox_from_patch_rows_cols(...)`


## 9. 反射率量纲与 nodata 处理更新

这是这次流程更新里最关键的部分。

### 9.1 S2 归一化

[geometry.py](../rtm_stagewise_sr/preprocessing/geometry.py) 中的 `normalize_s2(...)` 规则是：

- 如果 `max > 100`，则视为整型定标值，自动除以 `10000`
- 否则保持原样

### 9.2 HSI 归一化

同文件中的 `normalize_hsi(...)` 现在会：

1. 先转成 `float32`
2. 识别 `nodata`
3. 如果存在像 `-32768` 这样的整型编码 nodata，也会自动识别
4. 对有效值检查最大值
5. 如果有效最大值 `> 100`，自动除以 `10000`
6. 把 nodata 位置替换成 `0.0`

当前固定记录的缩放策略字符串是：

- `divide_by_10000_if_max_gt_100`

### 9.3 为什么要这样做

旧流程里：

- `PRISMA float32` 通常已经接近 `0~1 reflectance`
- `EnMAP int16` 更像是 `reflectance * 10000`

如果不先缩放：

- RTM teacher 会把绝大多数正值直接压到 `1`
- stagewise 输入量纲也会失真

现在这个问题已经在 onepatch 生成阶段被修正。


## 10. onepatch 元数据的新增字段

新的 `meta/ALL_POINTS.json` 里除了原来的 footprint、ROI、坐标信息之外，还会保存：

- `source_hsi_path`
- `source_hsi_scale_policy`
- `source_hsi_scale_factor`
- `source_hsi_nodata_value`
- `source_hsi_nodata_replaced_with_zero`
- `source_hsi_valid_min_after_scale`
- `source_hsi_valid_max_after_scale`

`diagnostics/dataset_summary.json` 里也会记录至少这两项：

- `source_hsi_scale_factor`
- `source_hsi_valid_max_after_scale`

这些字段现在是后续“是否允许复用 onepatch”的判定依据之一。


## 11. 诊断图与可视化的当前规则

### 11.1 HSI RGB 选择

诊断图和主可视化现在都优先按波长选：

- `R = 664.5 nm`
- `G = 560.0 nm`
- `B = 496.6 nm`

只有在 `wavelengths_path` 缺失时才回退到旧的固定 band index。

### 11.2 HSI 面板标题

可视化现在会根据源数据路径自动显示：

- `EnMAP RGB`
- `PRISMA RGB`
- 或 `HSI RGB`

### 11.3 nodata 拉伸修正

[visualization.py](../rtm_stagewise_sr/visualization.py) 中的 `_stretch(...)` 和 `_stretch_single(...)` 现在会忽略：

- 非有限值
- `<= -1e4` 的 nodata-like 值

这一步修正后，`EnMAP` 不会再因为 `-32768` 参与百分位统计而显示成接近黑白图。


## 12. RTM surrogate 的当前复用规则

### 12.1 基本逻辑

RTM 仍然是：

1. 根据目标 `wavelengths` 做 folding
2. 训练 `forward / inverse surrogate`
3. 导出 `TorchScript`
4. 在 teacher 和 stagewise anchor 中复用

### 12.2 共享位置

当前统一放在：

- `datasets_crop/_shared_rtm_surrogates/`

### 12.3 复用键

现在不再简单按“数据集名字”复用，而是按 `wavelengths` 指纹复用：

- 波段数
- 波长数组内容
- `md5` 摘要

这样做的效果是：

- 波长完全相同的数据集只训练一次 surrogate
- 尤其适合多个 `EnMAP single-date` 数据集共享同一套 `filtered_wavelengths`

PRISMA 和 EnMAP 现在使用同一个批处理入口和相同的 wavelength fingerprint 规则。


## 13. 当前批处理的复用与过期判定

### 13.1 onepatch 复用判定

批处理脚本现在不会无条件复用已有 `dataset_onepatch`。

它会检查：

- `meta/ALL_POINTS.json` 是否存在
- 其中记录的 `source_hsi_path` 是否和当前输入一致
- 对 `EnMAP` 来说，`source_hsi_scale_policy` 是否已经是
  `divide_by_10000_if_max_gt_100`

如果不满足，会自动判定为：

- `stale or legacy onepatch`

并重新生成。

### 13.2 run 结果复用判定

批处理脚本也不会无条件复用已有 `final_selection.json`。

现在会比较：

- `meta/ALL_POINTS.json` 的修改时间
- `final_selection.json` 的修改时间

如果 `meta_json` 比结果新，就会把旧 run 判为过期并忽略。

也就是说：

- 只要 onepatch 被重做过，老结果就不会被误认为“已完成”


## 14. 单数据集完整流程的当前顺序

[pipeline.py](../rtm_stagewise_sr/pipeline.py) 当前顺序如下。

### 14.1 Stage 1

- `Stage 1 no-RTM`

作用：

- 先学到空间结构稳定的无 RTM 超分结果

### 14.2 Prelude baseline

- `Prelude baseline veg_lowfreq_ms020_spec000`

作用：

- 构造一个更保守的 baseline
- 让最终方案在非关键谱段不过度偏移

### 14.3 Prelude teacher

默认 teacher 入口是：

- `teacher_run_tag = hs_only`

当前可选：

- `hs_only`
- `both`
- `ms_only`

输出目录示例：

- `prelude_teacher_hs_only_rtm/`

### 14.4 可选的 protected-band teacher 混合

如果指定 `teacher_protected_run_tag`，脚本还可以：

1. 额外训练一份 protected teacher
2. 只在指定波段范围内替换 base teacher
3. 生成 hybrid teacher

相关摘要会写到：

- `teacher_protected_band_blend_summary.json`

### 14.5 Final Stage 2

最终主线是：

- `Final scheme-3 winner teacher_veg_alpha_ms020_as010`

关键设置写在 README 里，核心包括：

- `stage2_update_mode = teacher_veg_band_alpha`
- `lambda_ms_stage2 = 0.20`
- `lambda_alpha_smooth = 0.01`
- `lambda_detail_lock = 1.0`
- `protect_band_nm = 700-1300`
- `blend_nm = 680-1320`

### 14.6 Post spectral smoothing

最终输出还会做一次分段光谱平滑：

- kernel 默认是 `(0.25, 0.5, 0.25)`
- 波段间隙大于 `30 nm` 时按 segment 分开平滑

摘要会写到：

- `post_smoothing_summary.json`


## 15. 单数据集输出内容

每次 run 会生成一个新的时间戳目录，例如：

```text
datasets_crop/<dataset_name>/runs_stagewise_centered_plot15aw_teacher_veg_alpha_ms020_as010_postsmooth_prismargb/<timestamp>_<run_label>/
```

其中核心文件包括：

- `train.log`
- `README.md`
- `run_inputs.json`
- `checkpoint_scores.csv`
- `best_checkpoint_info.json`
- `X_hat_hrhsi_result_raw.npy`
- `X_hat_hrhsi_result.npy`
- `final_field_metrics_raw.json`
- `final_field_metrics_700_1300_raw.json`
- `final_field_metrics.json`
- `final_field_metrics_700_1300.json`
- `spectra_40points_raw.png`
- `spectra_40points.png`
- `final_selection.json`

另外通常还会有：

- `stage1_visual_compare/`
- `stage2_visual_compare/`
- `stage2_visual_compare_raw/`
- `prelude_teacher_*_rtm/`
- `prelude_veg_lowfreq_ms020_spec000/`


## 16. batch 输出内容

PRISMA 和 EnMAP 使用统一的输出清单：

- `datasets_crop/batch_manifest.json`

manifest 中会记录：

- 数据集名
- dataset root
- generated root
- meta json
- wavelengths path
- surrogate tag
- surrogate dir
- output base
- completed run


## 17. 推荐执行方式

### 17.1 自动扫描全部 PRISMA 与 EnMAP 数据

```bash
python -m rtm_stagewise_sr.batch
```

### 17.2 只跑指定数据集

```bash
python -m rtm_stagewise_sr.batch dataset_a dataset_b
```

### 17.3 跑单个数据集

用：

- [pipeline.py](../rtm_stagewise_sr/pipeline.py)

把下面这些显式传进去：

- `--dataset_root`
- `--prisma_path`
- `--s2_path`
- `--meta_json`
- `--field_csv`
- `--wavelengths_path`
- `--rtm_inv_path`
- `--rtm_fwd_path`
- `--rtm_scalers_path`
- `--srf_xlsx`


## 18. 当前需要特别记住的兼容性说明

### 18.1 代码里仍有 `prisma_path` 这个历史命名

即使输入是 `EnMAP`，很多内部变量名和参数名仍然叫：

- `prisma_path`

这只是历史命名残留，不代表真的在用 PRISMA。

实际数据源请以 `meta/ALL_POINTS.json` 里的：

- `source_hsi_path`

为准。

### 18.2 旧结果不能和新量纲结果混用

如果旧 run 是在 `EnMAP` 未做反射率缩放之前生成的，那么这些结果应视为失效结果，不应与当前新版流程的结果直接比较。

### 18.3 可视化标题里的 `postsmooth_prismargb`

当前 run 目录名里仍保留了：

- `postsmooth_prismargb`

这是历史命名残留。

它不表示当前输入一定是 PRISMA，也不影响实际训练和评估流程。


## 19. 建议的后续维护原则

后续如果继续扩展数据类型，建议保持下面 4 条规则不变：

1. onepatch 生成阶段就完成量纲修正，不把传感器差异拖到训练阶段
2. 任何新传感器都要把 `source_hsi_*` 元数据写全
3. 任何可复用结果都要带“输入版本校验”或“时间戳过期判定”
4. 可视化层不要再写死 `PRISMA RGB`，统一以实际传感器路径自动推断


## 20. 结论

当前最新流程可以概括为：

- 几何上使用 centered + plot15 area-weighted onepatch
- 训练上使用 stagewise + RTM teacher + band-wise alpha fusion + post smoothing
- 数据输入上同时支持旧 `project_B4` PRISMA 流程和新根目录 `EnMAP single-date` 流程
- 量纲上显式修正 `EnMAP int16 -> reflectance`
- 复用策略上显式避免旧 onepatch 和旧 run 被误用

这份文档可以作为后续会话、后续批处理和后续代码清理的统一参考。

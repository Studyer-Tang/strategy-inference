# v0.7 多步接口性能记录

[原始 JSON](../benchmarks/results/multistep-0.7.json)绑定 v0.7.0 全部 Python 源码及 [runner](../benchmarks/multistep.py) 的 SHA-256。`git_head_at_run` 是计时前 HEAD，待发布工作树以逐文件哈希为准；旧版记录保持冻结。

## 测量条件

macOS 26.7.1 ARM64，Python 3.13.15，NumPy 2.5.3。四项任务各在独立进程中串行运行，预热一次，记录五次热调用并取中位数。五个 BLAS 线程环境变量设为 1；没有独立验证运行库实际线程数。

导入、输入生成、已有点预测和初始尺度、独立终态核验、指纹与调用后状态导出均不计入耗时。API 内部创建返回结果的成本计入。另作一次预热后的 `tracemalloc` 调用，包含新输出与临时分配，排除预先存在的输入及导入；此数值不是进程 RSS。

| 任务 | 观测数 × 步长数 | 热调用中位数 / ms | 跟踪分配峰值 / KiB |
| --- | --- | ---: | ---: |
| 批量 pooled，保存完整轨迹 | 12,000 × 4 | 183.005 | 4866.76 |
| 流式 pooled，保留当前状态 | 50,000 × 4 | 666.791 | 19.16 |
| 流式 pooled，保留当前状态 | 1,000 × 4 | 13.526 | 18.91 |
| 流式 interlaced，保留当前状态 | 50,000 × 4 | 678.160 | 24.98 |

物理步长为 1、6、12、24。输入是平稳 Gaussian AR(1)，phi=0.65，创新方差 1；点预测与初始尺度采用已知过程公式，仅用于固定工程负载。每时先 observe 再 predict，未来尾部不 flush，终态保留 43 个 pending。设置为 alpha=0.1、标量 step_size=0.1、decay=0.2、initial_quantile=0.65、scale_decay=0.97、scale_source="shortest"、scale_floor=1e−8。

性能负载的标量学习率与[正式统计研究](multistep-results.md)按步长设置的 `0.1/sqrt(h)` 不同。耗时、终态指纹和该工程输入的覆盖不用于证明尺度共享的统计收益。

每项任务以相同输入独立核验 class 与 wrapper 的完整终态，包括各 lane 阈值、成熟数、miss 数、默认尺度与 pending 区间。该核验在计时之外；五次采样及单独内存调用的结果指纹一致。

## 空间与适用方式

流式 class 不保留已成熟轨迹。连续发行时，未成熟队列的最大数量受 `sum(lead_times)` 控制；interlaced 还保存已用相位的状态。固定步长配置下，状态空间不随观测历史长度增长；步长增大时队列会增大，不能称为对所有参数都 O(1)。本轮两个 pooled 长度的跟踪峰值约 19 KiB，是实现检查的补充，不单凭两个测量点证明渐近结论。

批量 wrapper 保存每个起点与步长的数组，空间随 F×H 增长。需要持续运行而不保存完整历史时使用 class；需要绘图、评价和逐预测导出时使用 wrapper。`to_dict()` 构造序列化记录的额外成本不包含在流式调用计时里。

这些是当前机器的工程测量，没有与其他库进行同负载竞争性比较，不是跨硬件速度承诺。具体模型拟合成本、数据接入和序列化成本需另行测量。

## 复现

在 v0.7.0 checkout 中运行到新文件，runner 拒绝覆盖已有记录：

```bash
python benchmarks/multistep.py --output /tmp/multistep-benchmark.json
```

保存源码哈希、环境、全部五次采样与输入指纹后再比较改动。不同 Python、NumPy、平台和机器负载可能改变计时、分配及浮点指纹。

[v0.6 性能](time-series-performance.md)与 [v0.5 性能](performance.md)说明旧版本接口，不代替当前多步测量。

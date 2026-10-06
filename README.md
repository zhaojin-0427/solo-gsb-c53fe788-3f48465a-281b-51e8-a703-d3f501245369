# 色谱定量 HTTP 服务 (Chromatography Quantification API)

用 **Python / FastAPI / NumPy / SciPy (HiGHS)** 实现的在线色谱定量接口。
接收至多 6 种纯品（标准品）的时间-强度曲线、带编号的参考内标时刻、样本定量
通道曲线与内标通道上**无编号**的候选时刻（允许漏检与杂峰），通过

1. **锚点配对 DAG + 最长路 DP**：参考内标时刻与候选时刻保序配对、参考点和
   候选点均不复用、参考→样本的段斜率 `ds/dr` 必须落在漂移斜率范围内；
2. **共用分段线性校时** φ：把所有纯品模板插值到锚点之间的样本时间点；
3. **线性叠加 + 线性规划**：样本强度 = Σ rₖ · 模板ₖ，`rₖ = 样本浓度 / 纯品浓度 ≥ 0`；
   - 未饱和点：`|预测 − 观测| ≤ tolerance`
   - 饱和点（`intensity ≥ saturation`）：只需 `预测 ≥ saturation − tolerance`；
4. **先最大化可行配对数**，再在所有并列（tied）的最优解释上聚合每个组分的
   浓度**上下界**与锚点证据。重叠峰不强行唯一归属；无上界时返回 `null`。

返回的是**已证结果**：界是对所有可行最优解释的精确 LP 极值。

---

## 一键启动 (Docker Compose)

```bash
docker compose up --build
```

服务监听 http://localhost:8000

- 接口文档（Swagger UI）：http://localhost:8000/docs
- 健康检查：`GET http://localhost:8000/health`

本地直接运行（需要 Python 3.11+）：

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

测试与基准：

```bash
pip install -r requirements.txt pytest httpx
pytest                                  # 全部功能测试
pytest -m benchmark -s                  # 在线进样基准（见下文）
```

---

## 在线进样性能

`tests/test_quant.py::test_online_target` 固定为在线进样规模：

| 输入 | 规模 |
|---|---|
| 参考内标（编号） | **12** |
| 候选内标时刻（无编号，含杂峰） | **24** |
| 样本定量通道采样点 | **8192** |
| 纯品 | 4（上限 6） |

在普通开发机上约 **0.4 秒**返回完整证明的浓度区间（测试预算 10 秒，
`enumeration_truncated=false`）。大规模 LP 通过“按非零支撑分组的精确行约简
+ 2 维共流峰下半/上包络线”压缩到几十行，可行域与全量约束完全一致。

---

## 请求

`POST /quantify`，`Content-Type: application/json`

| 字段 | 类型 | 说明 |
|---|---|---|
| `standards` | 数组(1–6) | 纯品列表 |
| `standards[].id` | 整数 | 组分编号（可省，默认 1..n） |
| `standards[].concentration` | 正数 | 纯品浓度（必须 > 0） |
| `standards[].times` | 递增数值数组 | 纯品时间轴（严格递增） |
| `standards[].intensities` | 有限数值数组 | 纯品强度曲线（等长） |
| `reference_markers` | 非空数组 | **编号**参考内标时刻 |
| `reference_markers[].id` | 整数 | 内标编号（唯一） |
| `reference_markers[].time` | 数 | 参考时间（严格递增） |
| `sample.channel_id` | 字符串 | 可选，样本定量通道名 |
| `sample.times` | 递增数组 | 样本时间轴（严格递增） |
| `sample.intensities` | 有限数数组 | 样本强度（`≥ saturation` 视为饱和点） |
| `candidate_markers` | 非空数组 | 内标通道上**无编号**候选时刻，`{"time": x}` 或直接数字；自动排序去重 |
| `saturation` | 正数 | 检测器饱和值 |
| `tolerance` | 非负数 | 允许误差（未饱和点双向容差） |
| `drift_slope.min/max` | 正数 | 段斜率 `Δ样本时间/Δ参考时间` 允许范围（min>0 且 min≤max） |

### 与实现一致的请求示例

```json
{
  "standards": [
    {"id": 1, "concentration": 1.0,
     "times": [0.0, 0.25, 0.5, 0.75],
     "intensities": [0.0, 0.0, 0.0, 0.0]},
    {"id": 2, "concentration": 1.0,
     "times": [0.0, 0.25, 0.5, 0.75],
     "intensities": [0.0, 0.0, 0.0, 0.0]}
  ],
  "reference_markers": [{"id": 101, "time": 3.0}, {"id": 102, "time": 17.0}],
  "sample": {
    "channel_id": "FID-A",
    "times": [3.3, 3.641, 3.982],
    "intensities": [0.00504, 0.01535, 0.04102]
  },
  "candidate_markers": [{"time": 3.3}, {"time": 9.9}, {"time": 16.6}],
  "saturation": 1000.0,
  "tolerance": 0.05,
  "drift_slope": {"min": 0.8, "max": 1.2}
}
```

> 上面的强度只展示前 3 个点；完整可运行的最小例子（2 个高斯纯品 × 81 点、
> 样本 40 点、2 个参考内标、3 个候选时刻含 1 个杂峰 9.9）见仓库根目录
> `examples/tiny_request.json`，可用
> `curl -s localhost:8000/quantify -H 'content-type: application/json' \`
> `  --data @examples/tiny_request.json` 复现下面的响应。

### 响应示例（状态 `ok`，来自上面的最小例子，真实输出）

```json
{
  "status": "ok",
  "sample_channel": "FID-A",
  "max_pairs": 2,
  "longest_valid_matching": 2,
  "explanations_evaluated": 1,
  "matchings_of_that_length": 1,
  "enumeration_truncated": false,
  "components": [
    {"id": 1, "pure_concentration": 1.0,
     "ratio_lower": 1.99899027183312, "ratio_upper": 2.001012536347246,
     "concentration_lower": 1.99899027183312, "concentration_upper": 2.001012536347246},
    {"id": 2, "pure_concentration": 1.0,
     "ratio_lower": 1.4987367491604418, "ratio_upper": 1.5012645791476946,
     "concentration_lower": 1.4987367491604418, "concentration_upper": 1.5012645791476946}
  ],
  "anchor_pair_sets": [
    [
      {"ref_id": 101, "ref_time": 3.0, "sample_time": 3.3,
       "candidate_index": 0, "segment_slope_to_next": 0.9500000000000001},
      {"ref_id": 102, "ref_time": 17.0, "sample_time": 16.6,
       "candidate_index": 2}
    ]
  ],
  "anchor_evidence": [
    {"ref_id": 101, "ref_time": 3.0, "sample_time": 3.3,
     "candidate_index": 0, "explanations": 1, "support": 1.0},
    {"ref_id": 102, "ref_time": 17.0, "sample_time": 16.6,
     "candidate_index": 2, "explanations": 1, "support": 1.0}
  ],
  "elapsed_ms": 14.588
}
```

解读：真实配比 `r = (2.0, 1.5)`，容差 0.05 下被证在两个窄区间内；候选
`9.9` 是杂峰，未被任何解释使用（不出现在证据中）。

### 响应字段

| 字段 | 含义 |
|---|---|
| `status` | `ok` / `insufficient_anchors` / `infeasible` / `invalid` / `exhaustion` |
| `max_pairs` | 可行解释达到的最大锚点对数（至少 2） |
| `longest_valid_matching` | 忽略拟合可行性时 DAG 上最长的合法锚点匹配长度 |
| `explanations_evaluated` | 枚举到的可行（最优）解释数；这些解释彼此并列 |
| `matchings_of_that_length` | 该长度下所有锚点匹配数（含拟合不可行的） |
| `enumeration_truncated` | 是否因搜索预算（9 s / 50 万条路径）截断；`true` 时结果**未证** |
| `components[].ratio_lower/upper` | 样本/纯品浓度比 `rₖ` 的已证区间，上界 `null` = +∞ |
| `components[].concentration_lower/upper` | 浓度区间 = `rₖ × pure_concentration` |
| `anchor_pair_sets` | 每个可行解释使用的锚点对（含 `segment_slope_to_next`） |
| `anchor_evidence` | 锚点对在全部并列解释中出现的次数 `explanations` 与支持率 `support` |

### 状态码

- `200 ok`：定量成功，浓度区间与锚点证据为对所有可行最优解释的精确聚合；
  重叠峰表现为各组分非平凡区间，不强行唯一归属。
- `200 insufficient_anchors`：保序、不复用且段斜率全部达标的锚点对不足 2 对
  （内标可漏检/有杂峰，但合法锚点仍太少）。
- `200 infeasible`：锚点结构存在，但**没有任何** ≥2 对的解释能让未饱和点误差
  都不超过容差、且饱和点预测都达到 `saturation − tolerance`。
- `400 invalid`：非法数值（NaN/Infinity/非数值）、非正浓度、时间轴非严格递增、
  编号重复、斜率范围非法等；`errors` 数组逐条说明。
- `200 exhaustion`：搜索预算耗尽仍未枚举完（异常宽松的斜率/海量候选时），此时
  结果未证，响应中给出原因与建议；在线进样场景不会触发。

饱和语义：样本点 `intensity ≥ saturation` 视为饱和。饱和点只要求模型预测达到
`saturation − tolerance`（不设上界），因此仅被饱和点约束的组分其上界为
`null`（+∞），下界仍被认证。

---

## 方法要点

- **配对搜索**：候选锚点构成 DAG，弧 = 保序 + 不复用 + 段斜率∈[min,max]；
  最长路 DP 得最大对数，三维 DP 统计各长度匹配数，DFS 按长度从大到小枚举
  全部匹配（一旦某个长度出现可行解释即为最优长度，所有该长度可行解释并列）。
- **校时**：解释内共用同一条分段线性 φ（参考时间→样本时间），模板以
  `np.interp` 向量化重采样到样本网格；锚点跨度之外模板取 0、不参与约束。
- **约束与界**：每解释 1 次可行性 LP + 每组分上下界 LP（`r≥0`，HiGHS）。
  约束按模板非零支撑分组精确约简（单支撑取最紧、双支撑共流峰取包络线），
  对可行域完全等价（见 `test_envelope_reduction_is_exact`）。
- **预检**：仅使用**可证明必要**的单行/单列推论快速排除错误锚点解释——
  全零行要求 `|y| ≤ tolerance`；单支撑行把该组分比值夹到闭区间、逐组分求交，
  交集为空即不可行；饱和行必须含正元素。凡是不确定的匹配一律进入精确 LP
  裁决；不采用任何最小二乘解的残差做判据（L2 最优比值的逐点误差上界不能
  否定其他可行比值，例如模板三点全 1、样本 0/0/1 时最小二乘比值 1/3 误差
  2/3，而比值 0.5 的误差仅 0.5）。

## 目录结构

```
app/main.py        FastAPI 路由 (/quantify, /health, /)
app/quant.py       校验、锚点 DAG/DP、枚举、校时、行约简 LP 与聚合
tests/test_quant.py 功能、边界、饱和/重叠峰与 8192 点在线基准
examples/          可直接 curl 的请求样例
Dockerfile, docker-compose.yml, requirements.txt
```

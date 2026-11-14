# 漫散初态极限平滑器（纯后端，精确有理）

输入一个 UTF-8 JSON 模型，输出条件于全部非 `null` 观测的极限后验均值与
协方差系数。所有数值以整数分子 / 正整数分母的 `p/q` 字符串表示，全程使用
`fractions.Fraction`，不调用任何现成平滑器、数值阈值或符号极限引擎。

## 模型与待求量

```
x_0 ~ N(a, P + κ D Dᵀ),          κ → +∞
x_{t+1} = F x_t + w_t,           w_t ~ N(0, Q)
y_t = h x_t + v_t,               v_t ~ N(0, r)
```

噪声与初态互相独立；`y` 中的 `null` 表示该时刻未观测（不是零观测）。
将时刻 `0..T-1` 的状态按“先时刻、后分量”堆叠为 `X`，求

```
E[X | 观测]                      （长度 nT 的条件均值极限）
Cov(X | 观测) = κ A + B + o(1)   （两个 nT 阶系数矩阵，保留跨时刻协方差）
```

## 极限提取（自写条件化，不用有限 κ 替代）

拆分 `x_0 = a + e_0 + Dδ`，其中 `e_0 ~ N(0,P)`、`δ ~ N(0,κI_d)`，堆叠为

```
X = μ + W δ + η,   W_{t,*} = FᵗD,  μ_t = Fᵗa,  η ~ N(0, C0)
```

观测坐标写成 `z = Hμ + Uδ + ξ`（`U = WᵀHᵀ`，`ξ ~ N(0,S0)`，
`S0 = H C0 Hᵀ + rI` 严格正定）。对条件高斯做精确 Woodbury 展开后取 κ→∞，
令 `M = U S0⁻¹ Uᵀ`、`P₀` 为 range(M) 的正交投影、`M⁺` 为对称半正定矩阵 M
的 Moore–Penrose 逆、`K = W − E0 S0⁻¹ (HW)`（`E0 = C0 Hᵀ`）：

```
mean = μ + (E0 S0⁻¹ + K M⁺ U S0⁻¹)(z − Hμ)
A    = W (I − P₀) Wᵀ
B    = C0 − E0 S0⁻¹ E0ᵀ + K M⁺ Kᵀ
```

- `A` 承载**不可辨识的初始漂移方向**，非零即非零，不做置零或投影截断；
- 全缺测时退化为先验：`mean = μ`、`A = W Wᵀ`、`B = C0`；
- `d = 0` 时 `A` 为完整 nT 阶零矩阵，`B` 即普通有限卡尔曼平滑协方差；
- 投影与伪逆均通过精确有理的主元列基与普通逆构造，不含平方根。

## 目录

```
smoother/
  linalg.py    精确有理矩阵运算：Gauss-Jordan 求逆、主元列基、投影、PSD 检验
  validate.py  严格 JSON 校验（重复/未知成员、形状、满秩、对称半正定等）
  core.py      极限条件化、A/B 系数提取、p/q 序列化
  __main__.py  命令行入口（成功原子替换，失败保留旧文件）
run_smoother.py            顶层入口脚本
tests/acceptance.py        独立小模型验收（自带 float 参考实现）
examples/                  规格示例与含不可辨识漂移的示例
```

## 用法

### 一次性命令行（原入口，保持不变）

```bash
python3 -I run_smoother.py INPUT.json OUTPUT.json
# 或
python3 run_smoother.py INPUT.json OUTPUT.json
```

成功时 `OUTPUT.json` 仅含 `mean`、`A`、`B` 三个成员；失败时旧输出文件保持
不变，错误以 JSON 打到 stderr，形如
`{"error":{"path":"$.Q[1][0]","message":"..."}}`，路径为零基下标；
纯语法错误的路径为 `"$"`。

### 持久 HTTP 服务与独立工作进程（遥测补传）

补传区分**受理**与**平滑完成**：HTTP 服务只把修订原子写入 SQLite，绝不
参与计算；独立工作进程另行启动，计算再慢也不会阻塞修订受理。

```bash
python3 -I run_service.py init   data.db MODEL.json   # 合法原模型初始化为版本 0
python3 -I run_service.py serve  data.db --port 8080  # 持久 HTTP 服务
python3 -I run_worker.py         data.db              # 独立工作进程（可多个）
```

入口脚本会自行把仓库根目录加入路径，因此 `-I`（隔离模式）下也能运行；
也可在仓库根目录用 `python3 -m smoother.service ...`（不带 `-I`）触发同一
子命令。

`POST /revisions` 的请求体仅含 `id`、`changes`：

- `id` 为非空 ASCII 字母数字串；
- `changes` 为 `[零基时刻, 值]` 数组，时刻为非负整数（禁布尔、禁小数、
  禁负数），值沿用 `y` 的有理串或 `null` 规则；空数组合法；
- 同一请求内重复时刻、时刻越出模型长度一律拒绝。

先校验后写库，拒绝不改变任何状态。同 `id` 忽略数组项顺序、按数值比较：
内容相同返回原版本，内容不同返回 `409`；新 `id` 原子更新 `y`、版本加一，
响应体为整数版本号。非法或重复成员返回 `400`，错误体携带 JSON 路径与
零基下标（纯语法错误路径为 `$`）。

`GET /` 原子返回 `{"version", "published"}`：`version` 是当前输入版本；
`published` 初为 `null`，此后保留**上次完整发布**的快照（恰含 `version`、
`y`、`mean`、`A`、`B`，且与该版本同源；分数为约分字符串，分母 1 省略，
`y` 中的 `null` 保留）。

工作进程启动时领取一个持久递增的**代号（generation）**，使旧代号立即
失效；只有当前代号、且计算版本等于最新输入版本时，结果才会原子发布，
旧代号的结果与错误一律忽略。修订停止后当前进程必然完成最新版；进程
被强退后重启会从 SQLite 补算。

例：`y=[null]`，以 `id=a` 补入 `"2/4"` 后再以同 id 重试 `"1/2"`，两次
均返回版本 `1`。

## 目录

```
smoother/
  linalg.py    精确有理矩阵运算：Gauss-Jordan 求逆、主元列基、投影、PSD 检验
  validate.py  严格 JSON 校验（重复/未知成员、形状、满秩、对称半正定等）
  core.py      极限条件化、A/B 系数提取、p/q 序列化
  __main__.py  一次性命令行入口（成功原子替换，失败保留旧文件）
  service/
    schema.py      SQLite 模式、版本 0 引导、修订/代号/发布的原子事务
    revisions.py   修订请求的严格校验（id 字符集、时刻整数、重复/越界）
    server.py      只受理不计算的 HTTP 服务（ThreadingHTTPServer）
    worker.py      独立工作进程：代号领取、计算、按代号与版本原子发布
    crash.py       验收用强退钩子（默认关闭）
    __main__.py    init / serve / worker 子命令入口
run_smoother.py            一次性入口脚本
run_service.py             持久服务入口脚本
run_worker.py              工作进程入口脚本
tests/acceptance.py        独立小模型验收（自带 float 参考实现）
tests/service_acceptance.py 进程级服务验收（HTTP、接管、强退恢复、完整快照）
examples/                  规格示例与含不可辨识漂移的示例
```

## 验收

```bash
python3 -I tests/acceptance.py           # 数学与一次性 CLI
python3 -I tests/service_acceptance.py   # 持久服务、工作进程与崩溃恢复
```

第一个套件包含：规格示例的精确断言；不可辨识/部分辨识漂移与全缺测先验；
d=0 经典平滑；40 个随机小模型与**独立实现的有限 κ 浮点高斯条件化**对照，
验证 `Cov = κA + B + o(1)` 的均值收敛、割线斜率 A、截距 B 与 o(1) 衰减；
A、B 的对称半正定性；以及 CLI 的原子替换、失败保留与零基错误路径。

第二个套件以真实子进程驱动 HTTP 与 SQLite：版本 0 引导与 `2/4`→`1/2`
幂等例；全部非法/重复成员的 400 路径且拒绝不改状态；同 id 顺序无关的
数值相等与 409；空 changeset 与单调整数版本；代号接管使旧进程退出、
交错修订收敛到最新版完整快照；受理后回执前、发布提交前后的强退恢复；
以及工作进程被 SIGKILL 后重启补算最新版。

## 输入约束（任一违反即拒绝并给出 JSON 路径）

- 仅允许成员 `a,P,D,F,Q,h,r,y`，重复或未知成员一律拒绝；
- `a`、`h` 长度为 `n ≥ 1`，`P,F,Q` 为 n 阶，`D` 为 n×d 且列满秩
  （`0 ≤ d ≤ n`，`d=0` 用 n 个空数组）；
- `P,Q` 必须对称半正定，`r > 0`，`y` 非空且元素为有理串或 `null`；
- 所有数值必须是整数或 `p/q` 串，分母为正整数；裸 JSON 数字、布尔值、
  空白、小数均拒绝。

## 验收

```bash
python3 -I tests/acceptance.py
```

包含：规格示例的精确断言；不可辨识/部分辨识漂移与全缺测先验；d=0 经典
平滑；40 个随机小模型与**独立实现的有限 κ 浮点高斯条件化**对照，验证
`Cov = κA + B + o(1)` 的均值收敛、割线斜率 A、截距 B 与 o(1) 衰减；
A、B 的对称半正定性；以及 CLI 的原子替换、失败保留与零基错误路径。

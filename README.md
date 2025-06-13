# StatComp_final
这是一个南开大学统计与数据科学学院2025年统计计算的课程小组期末作业。
一个基于Python的交互式MCMC采样可视化工具，支持多种MCMC算法（HMC、NUTS、SMC等）在多峰正态分布上的采样过程可视化与结果分析。

## 功能特性
- 7种MCMC算法选择（HMC/NUTS/SMC等）
- 可配置采样参数（迭代次数、步长、链数量等）
- 可视化采样轨迹图
  - 采样点分布与真实分布对比
  - 链收敛性诊断（Rhat/ESS）
  - 不同算法ESS对比

## 环境要求
- Python 3.8+

## 安装步骤

### 1. 克隆仓库
```bash
git clone https://github.com/Jimmyzzt/StatComp_final.git
cd statcomp_final
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动网页

```bash
shiny run --reload app.py
```
> 只需要保证 `app.py` 和 `smc_sampler.py` 在同一文件夹下即可成功运行

应用启动后，会自动在终端显示访问地址（默认：http://127.0.0.1:8000），使用浏览器打开即可使用。

### 小组成员
杨东旭 张翕然 吴语涵 张哲滔

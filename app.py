import numpy as np
import pints
import matplotlib.pyplot as plt
import matplotlib as mpl
from shiny import App, reactive, render, ui
from shiny.types import FileInfo
import io
import base64
import time
import asyncio
from smc_sampler import smc_sampler

config = {
    "font.family":'serif',
    "font.serif": ['SimSun'],
    "mathtext.fontset":'stix',
    "font.sans-serif": ['Times New Roman'],  # 设置无衬线字体为 Times New Roman
    "axes.unicode_minus": False,  # 确保负号显示正确
    "font.size": 12,
}
plt.rcParams.update(config)
plt.switch_backend('Agg')  # 添加这行，切换为非交互式后端

# 定义多峰正态分布的对数似然类
class MultiModalNormalLogLikelihood(pints.LogPDF):
    def __init__(self, means, covariances, weights):
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.n_modes = len(means)
        self.dimensions = len(means[0])

    def n_parameters(self):
        return self.dimensions

    def __call__(self, x):
        log_likelihood = -np.inf
        for i in range(self.n_modes):
            diff = x - self.means[i]
            inv_cov = np.linalg.inv(self.covariances[i])
            det_cov = np.linalg.det(self.covariances[i])
            log_likelihood = np.logaddexp(
                log_likelihood,
                np.log(self.weights[i]) - 0.5 * np.dot(diff.T, np.dot(inv_cov, diff)) -
                0.5 * np.log(det_cov) - 0.5 * self.dimensions * np.log(2 * np.pi)
            )
        return log_likelihood

    def evaluateS1(self, x):
        log_likelihood = -np.inf
        gradient = np.zeros(self.dimensions)
        for i in range(self.n_modes):
            diff = x - self.means[i]
            inv_cov = np.linalg.inv(self.covariances[i])
            det_cov = np.linalg.det(self.covariances[i])
            log_term = np.log(self.weights[i]) - 0.5 * np.dot(diff.T, np.dot(inv_cov, diff)) - 0.5 * np.log(
                det_cov) - 0.5 * self.dimensions * np.log(2 * np.pi)
            if log_term > log_likelihood:
                log_likelihood = log_term
                gradient = -np.dot(inv_cov, diff)
        return log_likelihood, gradient


# 固定参数
dimensions = 2
means = [
    np.array([0.0, 0.0]),
    np.array([5.0, 5.0]),
    np.array([-5.0, 5.0]),
    np.array([5.0, -5.0]),
    np.array([-5.0, -5.0]),
]
covariances = [np.eye(dimensions) for _ in range(len(means))]
weights = [1.0 / len(means) for _ in range(len(means))]

# 创建对数似然和先验
log_likelihood = MultiModalNormalLogLikelihood(means, covariances, weights)
log_prior = pints.UniformLogPrior([-10] * dimensions, [10] * dimensions)
log_posterior = pints.LogPosterior(log_likelihood, log_prior)

# method_descriptions = {
#     "NoUTurnMCMC": (
#         "No-U-Turn Sampler (NUTS) 是一种自适应的哈密顿蒙特卡罗(HMC)变体。"
#         "它能自动调整模拟路径的长度，避免了手动设置leapfrog步数的需要，"
#         "通常对于复杂后验分布非常高效和稳健。"
#     ),
#     "RelativisticMCMC": (
#         "相对论MCMC (Relativistic HMC) 使用广义哈密顿动力学，引入一个类似“软”边界的势能项。"
#         "这有助于采样器在低概率区域探索，并可能更好地穿越不同模式之间的“能量壁垒”，但可能以牺牲局部采样效率为代价。"
#     ),
#     "MonomialGammaHamiltonianMCMC": (
#         "Monomial Gamma HMC 是一种HMC变体，它从伽马分布而不是标准的高斯分布中抽取动量。"
#         "这可以改变采样器的探索行为，对于具有特定几何形状（如重尾）的后验分布可能更有优势。"
#     ),
#     "MALAMCMC": (
#         "Metropolis-Adjusted Langevin Algorithm (MALA) 是一种利用后验梯度信息来指导提议的MCMC方法。"
#         "它通过模拟朗之万动力学产生候选点，然后用Metropolis-Hastings步骤进行修正，"
#         "通常比随机游走Metropolis-Hastings更高效。"
#     ),
#     "HamiltonianMCMC": (
#         "经典哈密顿蒙特卡罗 (Classic HMC) 模拟一个粒子在由负对数后验定义的势能场中的运动。"
#         "它利用梯度信息进行远距离、高接受率的移动，在处理高维和相关参数时非常有效，"
#         "但需要用户手动调整步长和步数。"
#     ),
#     "RDMC": (
#         "黎曼流形动力学蒙特卡罗 (Riemannian Dynamics Monte Carlo, RDMC) 的一种简化实现，"
#         "该方法在利用朗之万动力学进行提议的同时，根据接受率自适应地调整步长。"
#         "它旨在通过结合几何信息来提升采样效率，尤其适用于后验具有复杂局部结构的分布。"
#     ),
#     "SMC": (
#         "序贯蒙特卡罗 (Sequential Monte Carlo, SMC) 通过一系列重采样和变异步骤，"
#         "将一组代表先验分布的粒子（或样本）逐步转化为代表后验分布的粒子。"
#         "这种方法对于处理多模态分布和模型证据估计特别有效。"
#     )
# }

method_descriptions = {
    "HamiltonianMCMC": (
        "Hamiltonian Monte Carlo (HMC) 是一种结合哈密顿动力学与MCMC的高效采样算法，"
        "通过模拟物理系统中的能量守恒轨迹来探索高维参数空间，"
        "从而避免传统MCMC方法的低效问题，尤其适用于复杂、高维且强相关的分布。"
    ),
    "NoUTurnMCMC": (
        "No-U-Turn Sampler (NUTS) 是HMC的一种自适应扩展算法，"
        "旨在解决HMC中需要手动设置leapfrog步数的问题。"
        "NUTS不仅保留了HMC在高维空间中的高效性，且在多数情况下表现更优，"
        "通常对于复杂后验分布非常高效和稳健。"
    ),
    "MonomialGammaHamiltonianMCMC": (
        "Monomial Gamma Sampler (MGS) 是一种基于哈密顿动力学和辅助变量的高效采样算法，"
        "通过引入广义动能函数将HMC与切片采样统一起来。"
        "适用于处理复杂、高维且强相关的概率分布。"
    ),
    "RelativisticMCMC": (
        "Relativistic Monte Carlo (RMC) 是一种基于相对论动力学的改进型HMC算法，"
        "旨在解决传统HMC对时间离散化和动量分布尺度敏感的局限性。"
        "RMC通过引入相对论哈密顿量增强算法稳定性。"
    ),
    "SMC": (
        "Sequential Monte Carlo (SMC) 是一种灵活且通用的采样方法，"
        "通过序贯重要性采样和重采样逐步更新粒子权重，从而逼近目标分布。"
        "能够高效地探索复杂的目标分布，适用于贝叶斯推断、优化以及计算归一化常数比等问题。"
    ),
    "MALAMCMC": (
        "Metropolis-adjusted Langevin algorithm (MALA) 融合了Langevin动力学与Metropolis-Hastings框架，"
        "基于离散化的Langevin扩散方程生成候选样本，"
        "随后通过Metropolis-Hastings准则计算接受概率。"
        "相较于传统的随机游走Metropolis算法，MALA利用梯度信息显著提高了采样效率。"
    ),
    "RDMC": (
        "Reverse Diffusion Monte Carlo (RDMC) 是一种基于逆扩散过程的高效采样算法，"
        "将分数匹配问题转化为均值估计问题，避免了分数函数估计的复杂性。"
        "利用OU过程的显式解将目标分布扩散到标准正态分布，然后通过逆过程采样。"
        "在处理多峰和高维复杂分布时具有更高的效率，能显著减少样本相关性并提高收敛速度。"
    )
}

# UI界面
app_ui = ui.page_fluid(
    ui.tags.style(
        """
        .app-title {
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
        }
        .btn-run {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            width: 100%;
            margin-top: 10px;
        }
        .btn-run:hover {
            background-color: #45a049;
        }
        .param-slider {
            padding: 5px 0;
        }
        .plot-container {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            overflow: auto;  /* 添加滚动条以防万一 */
        }
        .method-description {
            background-color: #e9f5ff;
            padding: 12px;
            border-radius: 8px;
            margin-top: 10px;
            font-size: 0.9rem;
        }
        """
    ),
    ui.div(
        ui.h2("多峰分布采样可视化", class_="app-title"),
        class_="text-center"
    ),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("采样参数设置", class_="text-center"),
            ui.input_select(
                "method",
                "选择MCMC方法",
                choices={
                    "HamiltonianMCMC": "Hamiltonian Monte Carlo",
                    "NoUTurnMCMC": "No-U-Turn Sampler",
                    "MonomialGammaHamiltonianMCMC": "Monomial Gamma Sampler",
                    "RelativisticMCMC": "Relativistic Monte Carlo",
                    "SMC": "Sequential Monte Carlo",
                    "MALAMCMC": "MALA",
                    "RDMC": "Reverse Diffusion Monte Carlo",
                },
                selected="HamiltonianMCMC",
            ),
            ui.output_ui("method_desc"),

            # --- 原有参数（非SMC）---
            ui.panel_conditional(
                "input.method != 'SMC'",
                ui.input_numeric("iterations", "迭代次数", value=2000, min=500, max=10000),
                ui.input_numeric("warmup", "预热期", value=500, min=100, max=2000),
                ui.input_numeric("n_chains", "链的数量", value=3, min=1, max=5),
                ui.input_slider("step_size", "步长", min=0.01, max=1.0, value=0.1, step=0.01),
                ui.input_slider("mass", "质量矩阵", min=0.1, max=2.0, value=1.0, step=0.1),
            ),
            ui.panel_conditional(
                "input.method == 'SMC'",
                ui.input_numeric("n_chains_smc", "粒子数 (N)", value=500, min=100, max=2000),
                ui.input_numeric("iterations_smc", "MCMC迭代次数 (Neff)", value=150, min=50, max=500),
                ui.input_slider("burn_in_fraction", "MCMC预热比例", min=0.0, max=0.8, value=0.3, step=0.05),
            ),

            # --- 以下为旧写法 ---
            # ui.input_numeric("iterations", "迭代次数", value=2000, min=500, max=10000),
            # ui.input_numeric("warmup", "预热期", value=500, min=100, max=2000),
            # ui.input_numeric("n_chains", "链的数量", value=3, min=1, max=5),
            # ui.input_slider("step_size", "步长", min=0.01, max=1.0, value=0.1, step=0.01),
            # ui.input_slider("mass", "质量矩阵", min=0.1, max=2.0, value=1.0, step=0.1),

            ui.input_action_button("run", "运行采样", class_="btn-run"),
            width=300
        ),

        ui.navset_tab(
            ui.nav_panel("采样过程",
                         ui.row(
                             ui.column(6,
                                       ui.div(
                                           ui.h4("采样轨迹"),
                                           ui.output_plot("trace_plot", height="500px"),
                                           class_="plot-container"
                                       )
                                       ),
                             ui.column(6,
                                       ui.div(
                                           ui.h4("采样点分布"),
                                           ui.output_plot("scatter_plot", height="500px"),
                                           class_="plot-container"
                                       )
                                       )
                         )
                         ),
            ui.nav_panel("结果分析",
                         ui.row(
                             ui.column(6,
                                       ui.div(
                                           ui.h4("参数分布"),
                                           ui.output_plot("histogram_plot", height="500px"),
                                           class_="plot-container"
                                       )
                                       ),
                             ui.column(6,
                                       ui.div(
                                           ui.h4("链间比较"),
                                           ui.output_plot("chain_comparison", height="500px"),
                                           class_="plot-container"
                                       )
                                       )
                         ),
                         ui.row(
                             ui.column(12,
                                       ui.div(
                                           ui.h4("MCMC诊断统计"),
                                           # ui.output_text("diagnostics"),
                                            ui.output_text_verbatim("diagnostics"),
                                           class_="plot-container",
                                           style="height: 400px; overflow-y: scroll;"
                                       )
                                       )
                         )
                         ),
            ui.nav_panel("分布可视化",
                         ui.row(
                             ui.column(12,
                                       ui.div(
                                           ui.h4("多峰分布"),
                                           ui.output_plot("distribution_plot", height="600px"),
                                           class_="plot-container"
                                       )
                                       )
                         )
                         ),
            ui.nav_panel("算法比较",
                         ui.row(
                             ui.column(12,
                                       ui.div(
                                           ui.h4("不同方法比较"),
                                           ui.output_plot("comparison_plot", height="600px"),
                                           class_="plot-container"
                                       )
                                       )
                         )
                         )
        )
    )
)


# 服务器逻辑
def server(input, output, session):
    chains = reactive.Value(None)
    current_method = reactive.Value("")
    is_running = reactive.Value(False)
    results_history = reactive.Value([])

    # 方法描述输出
    @output
    @render.ui
    def method_desc():
        return ui.div(
            method_descriptions.get(input.method(), "无描述信息"),
            class_="method-description",
        )

    # 运行MCMC采样
    @reactive.Effect
    @reactive.event(input.run)
    async def run_mcmc():
        is_running.set(True)

        # 创建起始点
        xs = [np.random.uniform(-10, 10, dimensions) for _ in range(input.n_chains())]

        method_name = input.method()
        if method_name in ("RDMC", "SMC"):
            # 使用自定义算法
            if method_name == "RDMC":
                chains_result = await asyncio.to_thread(
                    rdmc_sampler,
                    log_posterior,
                    xs,
                    input.iterations(),
                    input.step_size(),
                )
            else:  # SMC分支
                # 从 log_prior 获取边界
                lower_bounds = [-10] * dimensions
                upper_bounds = [10] * dimensions

                chains_result = await asyncio.to_thread(
                    smc_sampler,
                    log_posterior,  # 对数后验对象
                    input.n_chains_smc(),  # 粒子数 (N)
                    input.iterations_smc(),  # MCMC迭代次数 (Neff)
                    lower_bounds,  # 新增：传递下界
                    upper_bounds,  # 新增：传递上界
                    input.burn_in_fraction()  # 新增：传递预热比例
                )
                print(chains_result.shape)
            # else:
            #     chains_result = await asyncio.to_thread(
            #         smc_sampler,
            #         log_posterior,
            #         input.n_chains(),
            #         input.iterations(),
            #     )
        else:
            # 获取方法类
            method_class = getattr(pints, method_name)

            mcmc = pints.MCMCController(
                log_posterior,
                input.n_chains(),
                xs,
                method=method_class,
            )
            mcmc.set_max_iterations(input.iterations())
            mcmc.set_log_to_screen(False)

            for sampler in mcmc.samplers():
                if hasattr(sampler, "set_leapfrog_step_size"):
                    sampler.set_leapfrog_step_size(input.step_size())
                if hasattr(sampler, "set_mass_matrix"):
                    mass_matrix = np.diag([input.mass()] * dimensions)
                    sampler.set_mass_matrix(mass_matrix)

            chains_result = await asyncio.to_thread(mcmc.run)

        chains_result = np.asarray(chains_result)
        chains.set(chains_result)
        current_method.set(method_name)
        # 保存ESS用于比较
        try:
            warmup = input.warmup()
            if current_method() == 'SMC':
                warmup = 0
            chains_for_ess = chains_result[:, warmup:, :]
            ess = pints.effective_sample_size(
                chains_for_ess.reshape(-1, chains_for_ess.shape[-1])
            )
            history = list(results_history() or [])
            history.append((method_name, ess))
            results_history.set(history[-5:])
        except Exception as e:
            print(f"计算ESS时发生错误: {e}")
        finally:
            is_running.set(False)

    # 绘制采样轨迹
    @output
    @render.plot
    def trace_plot():
        if chains() is None or is_running():
            return placeholder_plot("采样轨迹", "运行采样后显示结果")

        if current_method() != 'SMC':
            warmup = input.warmup()
            if current_method() != 'SMC':
                chains_value = chains()[:, warmup:, :]
            else:
                chains_value = chains()

            # 调整图像大小，增加高度并保持响应式
            fig, axes = plt.subplots(dimensions, 1, figsize=(10, 3 * dimensions), dpi=100)

            # 如果只有一维，将axes转换为列表
            if dimensions == 1:
                axes = [axes]

            for i in range(dimensions):
                for chain in chains_value:
                    axes[i].plot(chain[:, i], alpha=0.7)
                axes[i].set_title(f'参数 {i + 1} 的轨迹')
                axes[i].set_xlabel('迭代次数')
                axes[i].set_ylabel(f'')
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()
            return fig
        else:
            history = chains()
            warmup = 0
            history = history[:, warmup:, :]
            n_particles, n_iters, dim = history.shape

            if dim != 2:
                return placeholder_plot("采样轨迹", "仅支持2维参数空间的轨迹可视化")

            fig, ax = plt.subplots(figsize=(8, 6))
            selected_idx = np.linspace(0, n_particles - 1, 20, dtype=int)  # 只画10条轨迹

            for idx in selected_idx:
                traj = history[idx, :, :]  # shape: (n_iters, 2)
                ax.plot(traj[:, 0], traj[:, 1], alpha=0.7, lw=1)

            ax.set_title("代表性粒子轨迹（SMC）")
            ax.set_xlabel("维度 1")
            ax.set_ylabel("维度 2")
            ax.grid(alpha=0.3)
            return fig

    # 绘制采样点分布
    @output
    @render.plot
    def scatter_plot():
        if chains() is None or is_running():
            return placeholder_plot("采样点分布", "运行采样后显示结果")

        warmup = input.warmup()
        if current_method() != 'SMC':
            chains_value = chains()[:, warmup:, :]
        else:
            chains_value = chains()

        # 合并所有链
        all_points = np.vstack(chains_value)

        # 如果点太多，随机抽样一部分
        if len(all_points) > 5000:
            all_points = all_points[np.random.choice(len(all_points), 5000, replace=False)]

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

        # 绘制等高线背景
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(len(means)):
            diff = np.array([X, Y]).T - means[i]
            Z += weights[i] * np.exp(-0.5 * np.sum(diff @ np.linalg.inv(covariances[i]) * diff, axis=2))

        ax.contour(X, Y, Z, levels=10, colors='gray', alpha=0.3)

        # 绘制采样点
        ax.scatter(all_points[:, 0], all_points[:, 1],
                   s=5, alpha=0.4, c='blue',
                   label=f'{len(all_points)}个采样点')

        # 标记真实分布中心
        for i, mean in enumerate(means):
            ax.scatter(mean[0], mean[1], s=100, marker='*', c='red', edgecolor='black')

        ax.set_title(f'采样点分布 ({current_method()})')
        ax.set_xlabel('参数 1')
        ax.set_ylabel('参数 2')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.2)
        ax.legend()

        return fig

    # 绘制参数直方图
    @output
    @render.plot
    def histogram_plot():
        if chains() is None or is_running():
            return placeholder_plot("参数分布", "运行采样后显示结果")

        warmup = input.warmup()
        if current_method() != 'SMC':
            chains_value = chains()[:, warmup:, :]
        else:
            chains_value = chains()

        # 合并所有链
        all_points = np.vstack(chains_value)

        # 调整图像大小和子图间距
        fig, axes = plt.subplots(1, dimensions, figsize=(6 * dimensions, 4), dpi=100)
        fig.subplots_adjust(wspace=0.4)

        # 如果只有一维，将axes转换为列表
        if dimensions == 1:
            axes = [axes]

        # 创建图例的句柄
        legend_handles = []

        for i in range(dimensions):
            # 绘制直方图
            n, bins, patches = axes[i].hist(all_points[:, i], bins=30, density=True,
                                          alpha=0.7, color='skyblue', edgecolor='black')

            # 添加真实分布
            x = np.linspace(-10, 10, 200)
            true_pdf = np.zeros_like(x)

            for mode in range(len(means)):
                mean = means[mode][i]
                std = np.sqrt(covariances[mode][i, i])
                true_pdf += weights[mode] * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

            # 每个子图都添加
            line, = axes[i].plot(x, true_pdf, 'r-', linewidth=2)
            legend_handles.append(line)
            legend_handles.append(patches[0])

            axes[i].set_title(f'参数 {i + 1} 分布')
            axes[i].set_xlabel('')
            if i == 0:
                axes[i].set_ylabel('密度')
            else:
                axes[i].set_ylabel('')
            axes[i].grid(True, alpha=0.3)

        # 添加整体图例
        fig.legend(legend_handles, ['真实分布', '采样分布'],
                 loc='lower center', bbox_to_anchor=(0.5, -0.02),
                 ncol=2)

        plt.tight_layout()
        return fig

    @output
    @render.plot
    def chain_comparison():
        if chains() is None or is_running():
            return placeholder_plot("链间比较", "运行采样后显示结果")

        warmup = input.warmup()
        # 三维轨迹
        history = chains()  # (n_particles, total_iters, dim)
        particles, total_iters, dim = history.shape

        # 丢掉 warmup
        if current_method() != 'SMC':
            history = history[:, warmup:, :]  # -> (particles, iters_after_warmup, dim)
        iters_after = history.shape[1]

        fig, axes = plt.subplots(dim, 1, figsize=(10, 3 * dim), dpi=100)
        if dim == 1:
            axes = [axes]

        if current_method() == 'SMC':
            # 全局均值：每次迭代上所有粒子的平均
            global_means = history.mean(axis=0)  # -> (iters_after, dim)
            for i in range(dim):
                axes[i].plot(global_means[:, i], label='全局均值', linewidth=2)
                # 参考真实值（如果你有）
                true_mean = sum(w * m[i] for w, m in zip(weights, means))
                axes[i].axhline(true_mean, color='r', linestyle='--', label='真实均值')

                axes[i].set_title(f'参数 {i + 1} 全局均值收敛 (SMC)')
                axes[i].set_xlabel('迭代次数')
                axes[i].set_ylabel('均值')
                axes[i].legend()
                axes[i].grid(alpha=0.3)
        else:
            # MCMC 或多链：画每一条链的 running average
            for i in range(dim):
                for c in range(particles):
                    runavg = np.cumsum(history[c, :, i]) / np.arange(1, iters_after + 1)
                    axes[i].plot(runavg, alpha=0.6, label=f'链 {c + 1}' if c < 5 else None)
                true_mean = sum(w * m[i] for w, m in zip(weights, means))
                axes[i].axhline(true_mean, color='r', linestyle='--')
                axes[i].set_title(f'参数 {i + 1} 链收敛情况')
                axes[i].set_xlabel('迭代次数')
                axes[i].set_ylabel('均值')
                axes[i].grid(alpha=0.3)
                if i == 0:
                    axes[i].legend(ncol=2, fontsize='small')

        plt.tight_layout()
        return fig

    # 老版本链间比较图
    # @output
    # @render.plot
    # def chain_comparison():
    #     if chains() is None or is_running():
    #         return placeholder_plot("链间比较", "运行采样后显示结果")
    #
    #     warmup = input.warmup()
    #     chains_value = chains()[:, warmup:, :]
    #
    #     # 调整图像大小，高度根据参数数量自适应
    #     fig, axes = plt.subplots(dimensions, 1, figsize=(10, 3 * dimensions), dpi=100)
    #
    #     # 如果只有一维，将axes转换为列表
    #     if dimensions == 1:
    #         axes = [axes]
    #
    #     for i in range(dimensions):
    #         for chain_idx, chain in enumerate(chains_value):
    #             # 计算运行平均值
    #             running_avg = np.cumsum(chain[:, i]) / (np.arange(len(chain)) + 1)
    #             axes[i].plot(running_avg, label=f'链 {chain_idx + 1}', alpha=0.8)
    #
    #         # 添加真实分布的平均值（加权平均）
    #         true_mean = sum(weights[j] * means[j][i] for j in range(len(means)))
    #         axes[i].axhline(y=true_mean, color='r', linestyle='--', label='真实均值')
    #
    #         axes[i].set_title(f'参数 {i + 1} 的链收敛情况')
    #         axes[i].set_xlabel('迭代次数')
    #         axes[i].set_ylabel(f'参数 {i + 1} 平均值')
    #         axes[i].grid(True, alpha=0.3)
    #         axes[i].legend()
    #
    #     plt.tight_layout()
    #     return fig


    # 显示诊断信息
    @output
    @render.text
    def diagnostics():
        if chains() is None or is_running():
            return "请运行采样以查看诊断信息。"

        warmup = input.warmup()
        if current_method() != 'SMC':
            chains_value = chains()[:, warmup:, :]
        else:
            chains_value = chains()

        # 计算Rhat (Gelman-Rubin诊断)
        n_chains, n_iter, n_params = chains_value.shape
        rhats = []

        for param in range(n_params):
            # 计算每个链的平均值
            chain_means = np.mean(chains_value[:, :, param], axis=1)
            overall_mean = np.mean(chain_means)

            # 计算链间方差
            B = n_iter / (n_chains - 1) * np.sum((chain_means - overall_mean) ** 2)

            # 计算链内方差
            chain_vars = np.var(chains_value[:, :, param], axis=1, ddof=1)
            W = np.mean(chain_vars)

            # 估计后验方差
            var_plus = (n_iter - 1) / n_iter * W + B / n_iter

            # 计算Rhat
            rhat = np.sqrt(var_plus / W)
            rhats.append(rhat)

        # 计算有效样本大小 (ESS)
        ess = pints.effective_sample_size(
            chains_value.reshape(-1, chains_value.shape[-1])
        )

        # 构建诊断信息字符串
        diag_text = f"方法: {current_method()}\n"
        diag_text += f"迭代次数: {input.iterations()} (预热期: {input.warmup()})\n"
        diag_text += f"链数: {n_chains}\n"
        diag_text += f"步长: {input.step_size()}, 质量矩阵: {input.mass()}\n\n"

        diag_text += "参数诊断:\n"
        for i in range(n_params):
            diag_text += f"参数 {i + 1}:\n"
            diag_text += f"  Rhat: {rhats[i]:.4f} {'(良好 <1.1)' if rhats[i] < 1.1 else '(需关注 >1.1)'}\n"
            diag_text += f"  有效样本大小 (ESS): {ess[i]:.1f}\n"

        diag_text += "\n诊断说明:\n"
        diag_text += "- Rhat (Gelman-Rubin诊断): 衡量链收敛情况，小于1.1表示良好收敛\n"
        diag_text += "- ESS (有效样本大小): 表示独立样本的数量，越高越好\n"

        return diag_text

    # 绘制多峰分布
    @output
    @render.plot
    def distribution_plot():
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

        # 创建网格
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # 计算每个点的概率密度
        for i in range(len(means)):
            diff = np.array([X, Y]).T - means[i]
            Z += weights[i] * np.exp(-0.5 * np.sum(diff @ np.linalg.inv(covariances[i]) * diff, axis=2))

        # 绘制等高线图
        contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')

        # 标记峰值
        for i, mean in enumerate(means):
            ax.scatter(mean[0], mean[1], s=100, c='red', marker='*', edgecolor='black')

        ax.set_title('多峰正态分布')
        ax.set_xlabel('参数 1')
        ax.set_ylabel('参数 2')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.2)
        plt.colorbar(contour, ax=ax, label='概率密度')

        return fig

    # 绘制不同方法比较图
    @output
    @render.plot
    def comparison_plot():
        hist = results_history()
        if not hist:
            return placeholder_plot("算法比较", "运行采样后显示对比结果")

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        methods = [h[0] for h in hist]
        # ess_values = [np.mean(h[1], axis=0) for h in hist]
        ess_values = [np.atleast_1d(h[1]) for h in hist]  # 确保每个ESS值都是数组

        bar_width = 0.2 / dimensions
        indices = np.arange(len(methods))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, dimensions))

        for i in range(dimensions):
            vals = [ess[i] for ess in ess_values]
            ax.bar(indices + i * bar_width, vals, bar_width, label=f'参数 {i+1}', color=colors[i])

        ax.set_title('不同MCMC方法的有效样本大小(ESS)比较')
        ax.set_xlabel('MCMC方法')
        ax.set_ylabel('有效样本大小(ESS)')
        ax.set_xticks(indices + bar_width * (dimensions-1)/2)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        return fig

    # 生成占位图
def placeholder_plot(title, message):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, message,
            ha='center', va='center', fontsize=16,
            bbox=dict(facecolor='white', alpha=0.8))
    ax.set_title(title)
    ax.set_axis_off()
    return fig

# ------ 自定义采样算法 ------
def rdmc_sampler(log_post, starts, iterations, step_size):
    chains = []
    for x0 in starts:
        x = np.asarray(x0, dtype=float)
        samples = []
        acc = 0
        ss = step_size
        for i in range(iterations):
            # 梯度近似
            logp, grad = log_likelihood.evaluateS1(x)
            prop = x + ss * grad + np.sqrt(2 * ss) * np.random.randn(len(x))
            logp_prop, grad_prop = log_likelihood.evaluateS1(prop)
            log_accept = logp_prop - logp
            log_accept -= (np.sum((x - prop - ss * grad_prop) ** 2) - np.sum((prop - x - ss * grad) ** 2)) / (4 * ss)
            if np.log(np.random.rand()) < log_accept:
                x = prop
                acc += 1
            samples.append(x.copy())
            if (i + 1) % 100 == 0:
                rate = acc / 100.0
                if rate < 0.2:
                    ss *= 0.9
                elif rate > 0.8:
                    ss *= 1.1
                acc = 0
        chains.append(np.asarray(samples))
    return np.stack(chains)


# def smc_sampler(log_post, n_chains, iterations):
#     particles = np.random.uniform(-10, 10, (n_chains, iterations, dimensions))
#     # 简化实现：直接返回随机粒子序列
#     return particles



app = App(app_ui, server)
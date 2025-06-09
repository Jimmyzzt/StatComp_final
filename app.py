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

config = {
    "font.family":'serif',
    "font.serif": ['SimSun'],
    "mathtext.fontset":'stix',
    "font.sans-serif": ['Times New Roman'],  # 设置无衬线字体为 Times New Roman
    "axes.unicode_minus": False,  # 确保负号显示正确
    "font.size": 12,
}
plt.rcParams.update(config)

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

# MCMC方法描述
method_descriptions = {
    "NoUTurnMCMC": "No-U-Turn Sampler (NUTS) - 自适应HMC方法，不需要手动调整步长",
    "RelativisticMCMC": "相对论MCMC - 使用相对论动力学进行采样",
    "MonomialGammaHamiltonianMCMC": "Monomial Gamma HMC - 基于伽马分布的HMC变体",
    "MALAMCMC": "Metropolis Adjusted Langevin Algorithm - 使用梯度信息的MCMC方法",
    "HamiltonianMCMC": "经典Hamiltonian Monte Carlo - 使用物理模拟进行高效采样"
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
            ui.input_select("method", "选择MCMC方法",
                            choices={
                                "NoUTurnMCMC": "No-U-Turn MCMC",
                                "RelativisticMCMC": "相对论MCMC",
                                "MonomialGammaHamiltonianMCMC": "Monomial Gamma HMC",
                                "MALAMCMC": "MALA MCMC",
                                "HamiltonianMCMC": "Hamiltonian MCMC"
                            },
                            selected="NoUTurnMCMC"
                            ),
            ui.div(
                ui.p(method_descriptions["NoUTurnMCMC"], id="method_desc", class_="method-description")
            ),
            ui.input_numeric("iterations", "迭代次数", value=2000, min=500, max=10000),
            ui.input_numeric("warmup", "预热期", value=500, min=100, max=2000),
            ui.input_numeric("n_chains", "链的数量", value=3, min=1, max=5),
            ui.input_slider("step_size", "步长", min=0.01, max=1.0, value=0.1, step=0.01),
            ui.input_slider("mass", "质量矩阵", min=0.1, max=2.0, value=1.0, step=0.1),
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
                                           ui.output_text("diagnostics"),
                                           class_="plot-container",
                                           style="height: 200px; overflow-y: scroll;"
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

    # 更新方法描述 - 修复后的版本
    @reactive.Effect
    @reactive.event(input.method)
    def update_description():
        method = input.method()
        ui.update_text(id="method_desc", value=method_descriptions.get(method, "无描述信息"))

    # 运行MCMC采样
    @reactive.Effect
    @reactive.event(input.run)
    async def run_mcmc():
        is_running.set(True)

        # 创建起始点
        xs = [np.random.uniform(-10, 10, dimensions) for _ in range(input.n_chains())]

        # 获取方法类
        method_class = getattr(pints, input.method())

        # 创建MCMC控制器
        mcmc = pints.MCMCController(
            log_posterior,
            input.n_chains(),
            xs,
            method=method_class
        )

        # 设置参数
        mcmc.set_max_iterations(input.iterations())
        mcmc.set_log_to_screen(False)

        # 设置特定方法的参数
        for sampler in mcmc.samplers():
            if hasattr(sampler, 'set_leapfrog_step_size'):
                sampler.set_leapfrog_step_size(input.step_size())
            if hasattr(sampler, 'set_mass_matrix'):
                # 创建对角质量矩阵
                mass_matrix = np.diag([input.mass()] * dimensions)
                sampler.set_mass_matrix(mass_matrix)

        # 运行采样
        try:
            chains_result = await asyncio.to_thread(mcmc.run)
            chains.set(chains_result)
            current_method.set(input.method())
        except Exception as e:
            print(f"采样过程中发生错误: {e}")
        finally:
            is_running.set(False)

    # 绘制采样轨迹
    @output
    @render.plot
    def trace_plot():
        if chains() is None or is_running():
            return placeholder_plot("采样轨迹", "运行采样后显示结果")

        warmup = input.warmup()
        chains_value = chains()[:, warmup:, :]

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

    # 绘制采样点分布
    @output
    @render.plot
    def scatter_plot():
        if chains() is None or is_running():
            return placeholder_plot("采样点分布", "运行采样后显示结果")

        warmup = input.warmup()
        chains_value = chains()[:, warmup:, :]

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
        chains_value = chains()[:, warmup:, :]

        # 合并所有链
        all_points = np.vstack(chains_value)

        # 调整图像大小和子图间距
        fig, axes = plt.subplots(1, dimensions, figsize=(6 * dimensions, 4), dpi=100)
        plt.subplots_adjust(wspace=1)  # 增加子图之间的水平间距

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

            # 只在一个子图中绘制真实分布线并获取图例句柄
            if i == 0:
                line, = axes[i].plot(x, true_pdf, 'r-', linewidth=2)
                legend_handles.append(line)
                legend_handles.append(patches[0])

            axes[i].set_title(f'参数 {i + 1} 分布')
            axes[i].set_xlabel(f'')
            axes[i].set_ylabel('')
            axes[i].grid(True, alpha=0.3)

        # 添加整体图例
        fig.legend(legend_handles, ['真实分布', '采样分布'],
                 loc='lower center', bbox_to_anchor=(0.5, -0.02),
                 ncol=2)

        plt.tight_layout()
        return fig

    # 绘制链间比较图
    @output
    @render.plot
    def chain_comparison():
        if chains() is None or is_running():
            return placeholder_plot("链间比较", "运行采样后显示结果")

        warmup = input.warmup()
        chains_value = chains()[:, warmup:, :]

        # 调整图像大小，高度根据参数数量自适应
        fig, axes = plt.subplots(dimensions, 1, figsize=(10, 3 * dimensions), dpi=100)

        # 如果只有一维，将axes转换为列表
        if dimensions == 1:
            axes = [axes]

        for i in range(dimensions):
            for chain_idx, chain in enumerate(chains_value):
                # 计算运行平均值
                running_avg = np.cumsum(chain[:, i]) / (np.arange(len(chain)) + 1)
                axes[i].plot(running_avg, label=f'链 {chain_idx + 1}', alpha=0.8)

            # 添加真实分布的平均值（加权平均）
            true_mean = sum(weights[j] * means[j][i] for j in range(len(means)))
            axes[i].axhline(y=true_mean, color='r', linestyle='--', label='真实均值')

            axes[i].set_title(f'参数 {i + 1} 的链收敛情况')
            axes[i].set_xlabel('迭代次数')
            axes[i].set_ylabel(f'参数 {i + 1} 平均值')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        plt.tight_layout()
        return fig

    # 显示诊断信息
    @output
    @render.text
    def diagnostics():
        if chains() is None or is_running():
            return "请运行采样以查看诊断信息。"

        warmup = input.warmup()
        chains_value = chains()[:, warmup:, :]

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
        ess = pints.effective_sample_size(chains_value)

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
        # 这是一个静态图，用于比较不同方法
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        # 方法名称和颜色
        methods = ["NoUTurnMCMC", "HamiltonianMCMC", "MALAMCMC", "RelativisticMCMC"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # 假数据 - 在实际应用中应从真实运行中获取 牛逼
        ess_values = {
            "NoUTurnMCMC": [950, 980],
            "HamiltonianMCMC": [850, 870],
            "MALAMCMC": [780, 800],
            "RelativisticMCMC": [720, 750]
        }

        # 绘制条形图
        bar_width = 0.2
        indices = np.arange(len(methods))

        for i in range(dimensions):
            values = [ess_values[method][i] for method in methods]
            ax.bar(indices + i * bar_width, values, bar_width,
                   label=f'参数 {i + 1}', color=colors[i])

        ax.set_title('不同MCMC方法的有效样本大小(ESS)比较')
        ax.set_xlabel('MCMC方法')
        ax.set_ylabel('有效样本大小(ESS)')
        ax.set_xticks(indices + bar_width / 2)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        # 添加说明
        plt.figtext(0.5, 0.01,
                    "注: 此图为示例数据，实际结果可能因参数设置和随机性而有所不同。",
                    ha="center", fontsize=10, style='italic')

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


app = App(app_ui, server)
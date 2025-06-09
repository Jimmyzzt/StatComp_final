# --- START OF FILE app.py ---

import numpy as np
import pints
import matplotlib.pyplot as plt
import matplotlib as mpl
from shiny import App, reactive, render, ui
import asyncio
from collections import OrderedDict

# Matplotlib 配置
config = {
    "font.family": 'serif',
    "font.serif": ['SimSun', 'Times New Roman'],
    "mathtext.fontset": 'stix',
    "axes.unicode_minus": False,
    "font.size": 12,
}
plt.rcParams.update(config)


# 这个类定义将被传递给后台线程，所以它需要是全局的
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
            log_pdf_i = (
                    np.log(self.weights[i]) - 0.5 * np.dot(diff.T, np.dot(inv_cov, diff)) -
                    0.5 * np.log(det_cov) - 0.5 * self.dimensions * np.log(2 * np.pi)
            )
            log_likelihood = np.logaddexp(log_likelihood, log_pdf_i)
        return log_likelihood

    def evaluateS1(self, x):
        log_likelihood = -np.inf
        gradient = np.zeros(self.dimensions)
        for i in range(self.n_modes):
            diff = x - self.means[i]
            inv_cov = np.linalg.inv(self.covariances[i])
            det_cov = np.linalg.det(self.covariances[i])
            log_term = (
                    np.log(self.weights[i]) - 0.5 * np.dot(diff.T, np.dot(inv_cov, diff)) -
                    0.5 * np.log(det_cov) - 0.5 * self.dimensions * np.log(2 * np.pi)
            )
            if log_term > log_likelihood:
                log_likelihood = log_term
                gradient = -np.dot(inv_cov, diff)
        return log_likelihood, gradient


# 全局定义模型参数，这些是纯数据，可以安全地被后台线程访问
DIMENSIONS = 2
MEANS = [np.array([0.0, 0.0]), np.array([5.0, 5.0]), np.array([-5.0, 5.0]), np.array([5.0, -5.0]),
         np.array([-5.0, -5.0])]
COVARIANCES = [np.eye(DIMENSIONS) for _ in range(len(MEANS))]
WEIGHTS = [1.0 / len(MEANS) for _ in range(len(MEANS))]

# MCMC方法描述
method_descriptions = {
    "NoUTurnMCMC": "No-U-Turn Sampler (NUTS): HMC的一种自适应变体。它能自动调整模拟路径的长度，避免了手动设置“步数”这一棘手的参数，通常对于复杂后验分布非常高效和稳健。",
    "RelativisticMCMC": "Relativistic MCMC: HMC的一种变体，它使用相对论哈密顿动力学来生成提议。在某些能量壁垒较高的复杂分布中，它可能比标准HMC更能有效地探索参数空间。",
    "MonomialGammaHamiltonianMCMC": "Monomial Gamma HMC: 一种专门的HMC方法，它使用基于伽马分布的动能项。这对于采样具有特定约束（如参数为正）的分布可能特别有用。",
    "MALAMCMC": "Metropolis-Adjusted Langevin Algorithm (MALA): 一种利用梯度信息的MCMC方法。它使用朗之万动力学（梯度下降+噪声）来提议新的点，通常比随机游走Metropolis有更高的接受率和更快的收敛速度。",
    "HamiltonianMCMC": "Hamiltonian Monte Carlo (HMC): 经典的HMC算法。它通过模拟一个粒子在与目标分布的对数概率成比例的势能场中的运动来生成样本。这使得它可以提出距离很远且接受率很高的样本，对于高维问题非常高效，但需要仔细调整步长和步数。"
}

# 为主线程中的绘图函数创建一个 log_posterior 实例
# 这个实例专供 distribution_plot 和 scatter_plot 使用
log_likelihood_main = MultiModalNormalLogLikelihood(MEANS, COVARIANCES, WEIGHTS)
log_prior_main = pints.UniformLogPrior([-10] * DIMENSIONS, [10] * DIMENSIONS)
log_posterior_main = pints.LogPosterior(log_likelihood_main, log_prior_main)


# --- 线程安全的 MCMC 运行函数 ---
def run_mcmc_in_thread(method_name, n_chains, iterations, step_size, mass_val):
    """
    这个函数封装了所有pints MCMC操作，确保它们在同一个线程中创建和执行。
    """
    # 1. 在此线程内创建所有 Pints 对象
    log_likelihood = MultiModalNormalLogLikelihood(MEANS, COVARIANCES, WEIGHTS)
    log_prior = pints.UniformLogPrior([-10] * DIMENSIONS, [10] * DIMENSIONS)
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    # 2. 在此线程内生成初始点
    lower_bounds = [-10] * DIMENSIONS
    upper_bounds = [10] * DIMENSIONS
    initial_points = [np.random.uniform(low=lower_bounds, high=upper_bounds, size=DIMENSIONS) for _ in range(n_chains)]

    # 3. 在此线程内获取方法类并创建控制器
    method_class = getattr(pints, method_name)
    mcmc = pints.MCMCController(log_posterior, n_chains, initial_points, method=method_class)

    # 4. 在此线程内配置采样器
    mcmc.set_max_iterations(iterations)
    mcmc.set_log_to_screen(False)

    for sampler in mcmc.samplers():
        if hasattr(sampler, 'set_leapfrog_step_size'):
            sampler.set_leapfrog_step_size(step_size)
            if isinstance(sampler, pints.HamiltonianMCMC):
                sampler.set_leapfrog_steps(10)
        elif hasattr(sampler, 'set_step_size'):
            sampler.set_step_size(step_size)

        if hasattr(sampler, 'set_mass_matrix'):
            mass_matrix = np.diag([mass_val] * DIMENSIONS)
            sampler.set_mass_matrix(mass_matrix)

    # 5. 在此线程内运行 MCMC 并返回结果
    chains = mcmc.run()
    return chains


# UI界面
app_ui = ui.page_fluid(
    ui.tags.style(
        """
        .app-title { background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); color: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .card { border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; padding: 15px; background-color: #f8f9fa; }
        .btn-run { background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; width: 100%; margin-top: 10px; }
        .btn-run:hover { background-color: #45a049; }
        .plot-container { background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); overflow: auto; }
        .method-description { background-color: #e9f5ff; padding: 12px; border-radius: 8px; margin-top: 10px; font-size: 0.9rem; }
        """
    ),
    ui.div(ui.h2("多峰分布采样可视化", class_="app-title"), class_="text-center"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("采样参数设置", class_="text-center"),
            ui.input_select("method", "选择MCMC方法",
                            choices={
                                "NoUTurnMCMC": "No-U-Turn MCMC",
                                "RelativisticMCMC": "RelativisticMCMC",
                                "MonomialGammaHamiltonianMCMC": "Monomial Gamma HMC",
                                "MALAMCMC": "MALA MCMC",
                                "HamiltonianMCMC": "Hamiltonian MCMC"
                            },
                            selected="NoUTurnMCMC"),
            ui.div(ui.p(method_descriptions["NoUTurnMCMC"], id="method_desc", class_="method-description")),
            ui.input_numeric("iterations", "迭代次数", value=2000, min=500, max=10000),
            ui.input_numeric("warmup", "预热期", value=500, min=100, max=2000),
            ui.input_numeric("n_chains", "链的数量", value=3, min=1, max=5),
            ui.input_slider("step_size", "步长", min=0.01, max=1.0, value=0.1, step=0.01),
            ui.input_slider("mass", "质量矩阵(对角值)", min=0.1, max=2.0, value=1.0, step=0.1),
            ui.input_action_button("run", "运行采样", class_="btn-run"),
            width=300
        ),
        ui.navset_tab(
            ui.nav_panel("采样过程", ui.row(ui.column(6, ui.div(ui.h4("采样轨迹"),
                                                                ui.output_plot("trace_plot", height="500px"),
                                                                class_="plot-container")), ui.column(6, ui.div(
                ui.h4("采样点分布"), ui.output_plot("scatter_plot", height="500px"), class_="plot-container")))),
            ui.nav_panel("结果分析", ui.row(ui.column(6, ui.div(ui.h4("参数分布"),
                                                                ui.output_plot("histogram_plot", height="500px"),
                                                                class_="plot-container")), ui.column(6, ui.div(
                ui.h4("链间比较"), ui.output_plot("chain_comparison", height="500px"), class_="plot-container"))),
                         ui.row(ui.column(12, ui.div(ui.h4("MCMC诊断统计"), ui.output_text_verbatim("diagnostics"),
                                                     class_="plot-container",
                                                     style="height: 250px; overflow-y: scroll;")))),
            ui.nav_panel("分布可视化", ui.row(ui.column(12, ui.div(ui.h4("真实后验分布"),
                                                                   ui.output_plot("distribution_plot", height="600px"),
                                                                   class_="plot-container")))),
            ui.nav_panel("算法比较", ui.row(ui.column(12, ui.div(ui.h4("不同方法ESS比较"),
                                                                 ui.output_plot("comparison_plot", height="600px"),
                                                                 class_="plot-container"))))
        )
    )
)


# 服务器逻辑
def server(input, output, session):
    chains = reactive.Value(None)
    current_method = reactive.Value("")
    is_running = reactive.Value(False)
    comparison_data = reactive.Value(OrderedDict())

    @reactive.Effect
    @reactive.event(input.method)
    def update_description():
        ui.update_text("method_desc", value=method_descriptions.get(input.method(), "无描述信息"), session=session)

    @reactive.Effect
    @reactive.event(input.run)
    async def run_mcmc():
        if is_running.get(): return
        is_running.set(True)
        ui.update_action_button("run", label="正在运行...", disabled=True)
        chains.set(None)

        try:
            print(f"开始运行 {input.method()}...")
            # 调用完全隔离的后台任务
            chains_result = await asyncio.to_thread(
                run_mcmc_in_thread,
                input.method(),
                input.n_chains(),
                input.iterations(),
                input.step_size(),
                input.mass()
            )
            print("运行完成!")

            # 以下是在主线程中安全地处理返回的数据
            warmup = input.warmup()
            if input.iterations() > warmup:
                chains_after_warmup = chains_result[:, warmup:, :]
                if chains_after_warmup.shape[1] > 0:
                    ess = pints.effective_sample_size(chains_after_warmup)
                    new_data = comparison_data.get().copy()
                    new_data[input.method()] = ess
                    while len(new_data) > 5:
                        new_data.popitem(last=False)
                    comparison_data.set(new_data)

            chains.set(chains_result)
            current_method.set(input.method())

        except Exception as e:
            m = ui.modal(f"采样过程中发生错误: {e}", title="运行失败", easy_close=True, footer=None)
            ui.modal_show(m, session=session)
            print(f"采样过程中发生错误: {e}")
        finally:
            is_running.set(False)
            ui.update_action_button("run", label="运行采样", disabled=False)

    def placeholder_plot(title, message):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="lightsteelblue", lw=2))
        ax.set_title(title, fontsize=20, pad=20)
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.spines['top'].set_visible(False);
        ax.spines['right'].set_visible(False);
        ax.spines['bottom'].set_visible(False);
        ax.spines['left'].set_visible(False)
        return fig

    @output
    @render.plot
    def trace_plot():
        if chains() is None: return placeholder_plot("采样轨迹", "运行采样后显示结果")
        if is_running(): return placeholder_plot("采样轨迹", "正在采样，请稍候...")
        warmup = input.warmup()
        chains_value = chains()[:, warmup:, :]
        fig, axes = plt.subplots(DIMENSIONS, 1, figsize=(10, 3 * DIMENSIONS), dpi=100, sharex=True)
        if DIMENSIONS == 1: axes = [axes]
        for i in range(DIMENSIONS):
            for chain in chains_value:
                axes[i].plot(chain[:, i], alpha=0.7)
            axes[i].set_ylabel(f'参数 {i + 1}')
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel('迭代次数 (预热后)')
        fig.suptitle(f'{current_method()} 轨迹图', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    @output
    @render.plot
    def scatter_plot():
        if chains() is None: return placeholder_plot("采样点分布", "运行采样后显示结果")
        if is_running(): return placeholder_plot("采样点分布", "正在采样，请稍候...")
        warmup = input.warmup()
        chains_value = chains()[:, warmup:, :]
        all_points = np.vstack(chains_value)
        if len(all_points) > 5000:
            all_points = all_points[np.random.choice(len(all_points), 5000, replace=False)]
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        x_grid = np.linspace(-10, 10, 100)
        y_grid = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        pos = np.c_[X.ravel(), Y.ravel()]
        log_z = [log_posterior_main(p) for p in pos]
        Z = np.exp(np.array(log_z)).reshape(X.shape)
        ax.contour(X, Y, Z, levels=10, colors='gray', alpha=0.5)
        ax.scatter(all_points[:, 0], all_points[:, 1], s=5, alpha=0.4, c='blue', label=f'{len(all_points)}个采样点')
        for i, mean in enumerate(MEANS):
            ax.scatter(mean[0], mean[1], s=100, marker='*', c='red', edgecolor='black',
                       label="真实模式中心" if i == 0 else "")
        ax.set_title(f'采样点分布 ({current_method()})')
        ax.set_xlabel('参数 1');
        ax.set_ylabel('参数 2')
        ax.set_xlim(-10, 10);
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.2)
        ax.legend();
        ax.set_aspect('equal', adjustable='box')
        return fig

    @output
    @render.plot
    def histogram_plot():
        if chains() is None: return placeholder_plot("参数分布", "运行采样后显示结果")
        if is_running(): return placeholder_plot("参数分布", "正在采样，请稍候...")
        warmup = input.warmup()
        chains_value = chains()[:, warmup:, :]
        all_points = np.vstack(chains_value)
        fig, axes = plt.subplots(1, DIMENSIONS, figsize=(5 * DIMENSIONS, 5), dpi=100, sharey=True)
        if DIMENSIONS == 1: axes = [axes]
        for i in range(DIMENSIONS):
            axes[i].hist(all_points[:, i], bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black',
                         label='采样分布')
            x_grid = np.linspace(-10, 10, 200)
            true_pdf = np.zeros_like(x_grid)
            for mode in range(len(MEANS)):
                mean = MEANS[mode][i]
                std = np.sqrt(COVARIANCES[mode][i, i])
                true_pdf += WEIGHTS[mode] * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
                    -0.5 * ((x_grid - mean) / std) ** 2)
            axes[i].plot(x_grid, true_pdf, 'r-', linewidth=2, label='真实边际分布')
            axes[i].set_title(f'参数 {i + 1} 分布')
            axes[i].set_xlabel(f'参数 {i + 1} 值')
            axes[i].grid(True, alpha=0.3)
        axes[0].set_ylabel('概率密度')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2, wspace=0.1)
        return fig

    @output
    @render.plot
    def chain_comparison():
        if chains() is None: return placeholder_plot("链间比较", "运行采样后显示结果")
        if is_running(): return placeholder_plot("链间比较", "正在采样，请稍候...")
        warmup = input.warmup()
        chains_value = chains()[:, warmup:, :]
        fig, axes = plt.subplots(DIMENSIONS, 1, figsize=(10, 3 * DIMENSIONS), dpi=100, sharex=True)
        if DIMENSIONS == 1: axes = [axes]
        for i in range(DIMENSIONS):
            for chain_idx, chain in enumerate(chains_value):
                running_avg = np.cumsum(chain[:, i]) / (np.arange(len(chain)) + 1)
                axes[i].plot(running_avg, label=f'链 {chain_idx + 1}', alpha=0.8)
            true_mean = sum(WEIGHTS[j] * MEANS[j][i] for j in range(len(MEANS)))
            axes[i].axhline(y=true_mean, color='r', linestyle='--', label='真实均值')
            axes[i].set_ylabel(f'参数 {i + 1} 均值')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        axes[-1].set_xlabel('迭代次数 (预热后)')
        fig.suptitle('各链运行平均值的收敛情况', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    @output
    @render.text
    def diagnostics():
        if chains() is None:
            if is_running(): return "正在采样，请稍候..."
            return "请运行采样以查看诊断信息。"
        warmup = input.warmup()
        chains_value = chains()[:, warmup:, :]
        if chains_value.shape[1] < 2: return "采样点过少（预热后），无法计算诊断统计。"
        rhats = pints.rhat(chains_value)
        ess = pints.effective_sample_size(chains_value)
        diag_text = f"方法: {current_method()}\n"
        diag_text += f"总迭代: {input.iterations()}, 预热期: {input.warmup()}, 采样数: {input.iterations() - warmup}\n"
        diag_text += f"链数: {input.n_chains()}\n"
        diag_text += f"步长: {input.step_size()}, 质量: {input.mass()}\n"
        diag_text += "----------------------------------------\n"
        diag_text += "参数诊断:\n"
        for i in range(DIMENSIONS):
            diag_text += f"  参数 {i + 1}:\n"
            diag_text += f"    R-hat: {rhats[i]:.4f} {'(收敛良好 < 1.05)' if rhats[i] < 1.05 else '(可能未收敛 > 1.05)'}\n"
            diag_text += f"    有效样本大小 (ESS): {ess[i]:.1f}\n"
        diag_text += "----------------------------------------\n"
        diag_text += "诊断说明:\n- R-hat (Gelman-Rubin): 衡量多条链的一致性。接近1.0表示链已混合且收敛。\n- ESS: 衡量采样效率。表示与当前采样结果方差相当的独立样本数量，越高越好。\n"
        return diag_text

    @output
    @render.plot
    def distribution_plot():
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        x_grid = np.linspace(-10, 10, 150)
        y_grid = np.linspace(-10, 10, 150)
        X, Y = np.meshgrid(x_grid, y_grid)
        pos = np.c_[X.ravel(), Y.ravel()]
        log_z_values = [log_posterior_main(p) for p in pos]
        Z = np.exp(np.array(log_z_values)).reshape(X.shape)
        contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
        ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.2)
        for i, mean in enumerate(MEANS):
            ax.scatter(mean[0], mean[1], s=100, c='red', marker='*', edgecolor='black')
        ax.set_title('真实后验概率分布')
        ax.set_xlabel('参数 1');
        ax.set_ylabel('参数 2')
        ax.set_xlim(-10, 10);
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.2);
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar(contour, ax=ax, label='概率密度')
        return fig

    @output
    @render.plot
    def comparison_plot():
        data = comparison_data.get()
        if not data: return placeholder_plot("不同方法ESS比较", "请先运行至少一种采样方法")
        methods = list(data.keys())
        ess_values = list(data.values())
        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        n_methods = len(methods)
        indices = np.arange(n_methods)
        bar_width = 0.35
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, DIMENSIONS))
        for i in range(DIMENSIONS):
            values = [ess[i] for ess in ess_values]
            pos = indices - (bar_width * (DIMENSIONS - 1) / 2) + (i * bar_width)
            ax.bar(pos, values, bar_width, label=f'参数 {i + 1} ESS', color=colors[i])
        ax.set_title('不同MCMC方法的有效样本大小(ESS)比较')
        ax.set_xlabel('MCMC方法')
        ax.set_ylabel('有效样本大小 (ESS)')
        ax.set_xticks(indices)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        return fig


app = App(app_ui, server)

# --- END OF FILE app.py ---
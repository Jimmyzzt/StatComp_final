import numpy as np
from collections import namedtuple


# --- 辅助函数 ---

def deterministicR(inIndex, q):
    """确定性重采样算法（Kitagawa方法）"""
    n_chains = inIndex.shape[0]
    parents = np.arange(n_chains)
    N_childs = np.zeros(n_chains, dtype=int)

    cum_dist = np.cumsum(q)
    # --- 关键修复 ---
    # 强制最后一个元素为1.0，防止浮点数误差导致越界
    cum_dist[-1] = 1.0

    aux = np.random.rand(1)
    u = (parents + aux) / n_chains
    j = 0
    for i in parents:
        while u[i] > cum_dist[j]:
            j += 1
        N_childs[j] += 1

    indx = 0
    outindx = np.zeros(n_chains, dtype=int)
    for i in parents:
        if N_childs[i] > 0:
            outindx[indx:indx + N_childs[i]] = parents[i]
            indx += N_childs[i]
    return outindx


def AMH(X, target, covariance, mrun, beta, LB, UB, burn_in=0):
    """
    自适应Metropolis-Hastings采样器。
    返回完整的MCMC链历史。
    """
    Dims = covariance.shape[0]
    burn_in_steps = int(mrun * burn_in)
    total_steps = mrun + burn_in_steps

    chain_history = np.zeros((total_steps, Dims))
    posterior_history = np.zeros(total_steps)

    U = np.log(np.random.rand(total_steps))
    logpdf = target(X)
    V = covariance
    P0 = logpdf * beta

    a, b = 1 / 9, 8 / 9
    accepted_count = 0

    def dimension(i):
        switcher = {1: 0.441, 2: 0.352, 3: 0.316, 4: 0.285, 5: 0.275, 6: 0.273, 7: 0.270, 8: 0.268, 9: 0.267, 10: 0.266,
                    11: 0.265, 12: 0.255}
        return switcher.get(i, 0.255)

    s = a + b * dimension(Dims)
    current_X = np.copy(X)

    for i in range(total_steps):
        X_new = np.random.multivariate_normal(current_X, s ** 2 * V)

        # 边界处理 (反射边界)
        ind1 = np.where(X_new < LB)
        if len(ind1[0]) > 0:
            X_new[ind1] = 2 * LB[ind1] - X_new[ind1]

        ind2 = np.where(X_new > UB)
        if len(ind2[0]) > 0:
            X_new[ind2] = 2 * UB[ind2] - X_new[ind2]

        # 再次检查，防止双重反射出界
        X_new = np.maximum(X_new, LB)
        X_new = np.minimum(X_new, UB)

        P_new = beta * target(X_new)

        # 接受/拒绝步骤
        rho = P_new - P0
        if U[i] <= rho:
            current_X = X_new
            P0 = P_new
            if i >= burn_in_steps:
                accepted_count += 1

        chain_history[i, :] = current_X
        posterior_history[i] = P0

    avg_acc = accepted_count / mrun if mrun > 0 else 0

    final_chain = chain_history[burn_in_steps:, :]
    final_posterior = posterior_history[burn_in_steps:]

    return final_chain, final_posterior / beta if beta > 0 else final_posterior, avg_acc


# --- SMC 核心类 ---

class _SMCclass:
    """SMC采样核心类 (内部使用)"""

    def __init__(self, opt, samples, NT2, verbose=True):
        self.verbose = verbose
        self.opt = opt
        self.samples = samples
        self.NT2 = NT2

    def initialize(self):
        if self.verbose:
            print(f"Initializing SMC with {self.opt.N} chains and {self.opt.Neff} iterations per stage.")

    def prior_samples(self):
        """生成先验样本"""
        numpars = self.opt.LB.shape[0]
        diffbnd = self.opt.UB - self.opt.LB
        sampzero = self.opt.LB + np.random.rand(self.opt.N, numpars) * diffbnd
        postval = np.array([self.opt.target(x) for x in sampzero])[:, None]
        return self.NT2(sampzero, postval, np.array([0.0]), np.array([1]), None, None)

    def find_beta(self):
        """计算下一阶段的 beta 参数"""
        beta1 = self.samples.beta[-1]
        max_post = np.max(self.samples.postval)
        logpst = self.samples.postval - max_post
        beta = min(1.0, beta1 + 0.5)

        refcov = 1.0

        lower_bound, upper_bound = beta1, beta

        while upper_bound - lower_bound > 1e-6:
            curr_beta = (upper_bound + lower_bound) / 2
            # --- 逻辑修复 ---
            # 权重应由 beta 的增量决定
            delta_beta = curr_beta - self.samples.beta[-1]
            logwght = delta_beta * logpst
            wght = np.exp(logwght)
            covwght = np.std(wght) / np.mean(wght)

            if covwght > refcov:
                upper_bound = curr_beta
            else:
                lower_bound = curr_beta

        betanew = min(1.0, upper_bound)
        betaarray = np.append(self.samples.beta, betanew)
        newstage = np.append(self.samples.stage, self.samples.stage[-1] + 1)
        return self.NT2(self.samples.allsamples, self.samples.postval,
                        betaarray, newstage, self.samples.covsmpl,
                        self.samples.resmpl)

    def resample_stage(self):
        """重采样模型样本"""
        logpst = self.samples.postval - np.max(self.samples.postval)
        logwght = (self.samples.beta[-1] - self.samples.beta[-2]) * logpst.flatten()
        wght = np.exp(logwght)

        probwght = wght / np.sum(wght)
        inind = np.arange(self.opt.N)

        outind = deterministicR(inind, probwght)
        newsmpl = self.samples.allsamples[outind, :]
        newpostval = self.samples.postval[outind]  # 重采样后，后验值也需要同步更新

        return self.NT2(self.samples.allsamples, newpostval,  # 使用新的后验值
                        self.samples.beta, self.samples.stage,
                        self.samples.covsmpl, newsmpl)

    def make_covariance(self):
        """根据前一阶段的权重和样本计算模型协方差"""
        dims = self.samples.allsamples.shape[1]
        logpst = self.samples.postval - np.max(self.samples.postval)
        logwght = (self.samples.beta[-1] - self.samples.beta[-2]) * logpst.flatten()
        wght = np.exp(logwght)

        probwght = wght / np.sum(wght)

        meansmpl = np.sum(self.samples.allsamples * probwght[:, None], axis=0)

        smpldiff = self.samples.allsamples - meansmpl
        covariance = np.dot(smpldiff.T, smpldiff * probwght[:, None])

        return self.NT2(self.samples.allsamples, self.samples.postval,
                        self.samples.beta, self.samples.stage,
                        covariance, self.samples.resmpl)

    def MCMC_samples(self, burn_in=0):
        """执行MCMC采样"""
        dims = self.samples.allsamples.shape[1]
        mhsmpl_history = np.zeros([self.opt.N, self.opt.Neff, dims])
        mhpost_history = np.zeros([self.opt.N, self.opt.Neff])

        total_acc_rate = 0.0

        for i in range(self.opt.N):
            start = self.samples.resmpl[i, :]
            chain, posterior, acc_rate = AMH(
                start, self.opt.target, self.samples.covsmpl,
                self.opt.Neff, self.samples.beta[-1], self.opt.LB, self.opt.UB,
                burn_in=burn_in
            )
            mhsmpl_history[i, :, :] = chain
            mhpost_history[i, :] = posterior
            total_acc_rate += acc_rate

        avg_acc = total_acc_rate / self.opt.N
        if self.verbose:
            print(
                f"Stage {self.samples.stage[-1]}: Beta = {self.samples.beta[-1]:.4f}, Avg. Acceptance Rate = {avg_acc:.3f}")

        # 在最后阶段，allsamples 更新为完整的 MCMC 历史
        # 在中间阶段，allsamples 更新为每个粒子的最终位置
        if abs(self.samples.beta[-1] - 1.0) < 1e-6:
            final_samples_for_stage = mhsmpl_history
        else:
            final_samples_for_stage = mhsmpl_history[:, -1, :]

        return self.NT2(final_samples_for_stage, mhpost_history[:, -1], self.samples.beta,
                        self.samples.stage, self.samples.covsmpl, self.samples.resmpl)


# --- 主接口函数 (内部使用) ---

def _SMC_samples(opt, initial_samples=None, burn_in=0, verbose=True):
    """SMC 核心逻辑 (内部使用)"""
    NT2 = namedtuple('Samples', ['allsamples', 'postval', 'beta', 'stage', 'covsmpl', 'resmpl'])

    samples = initial_samples if initial_samples is not None else NT2(None, None, None, None, None, None)

    current = _SMCclass(opt, samples, NT2, verbose=verbose)
    current.initialize()

    if current.samples.allsamples is None:
        samples = current.prior_samples()

    while samples.beta[-1] < 1.0 - 1e-6:
        current.samples = samples
        samples = current.find_beta()

        current.samples = samples
        samples = current.resample_stage()

        current.samples = samples
        samples = current.make_covariance()

        current.samples = samples
        samples = current.MCMC_samples(burn_in=burn_in)

    return samples


# --- 公开 API ---

def smc_sampler(log_posterior, n_chains, iterations, lower_bounds, upper_bounds, burn_in_fraction=0.2):
    """
    使用序贯蒙特卡罗（SMC）对目标后验分布进行采样。

    参数:
        log_posterior: pints.LogPosterior 对象或类似的 callable，返回对数后验概率。
        n_chains (int): 采样链（粒子）的数量。
        iterations (int): 每个阶段（如果需要）或最终MCMC的迭代次数。
        lower_bounds (np.array): 参数的下界。
        upper_bounds (np.array): 参数的上界。
        burn_in_fraction (float): MCMC 阶段的预热比例。

    返回:
        np.array: 形状为 (n_chains, iterations, dimensions) 的样本数组。
    """
    Opt = namedtuple('Opt', ['target', 'LB', 'UB', 'N', 'Neff'])

    opt = Opt(
        target=log_posterior,
        LB=np.array(lower_bounds),
        UB=np.array(upper_bounds),
        N=n_chains,
        Neff=iterations
    )

    final_samples_obj = _SMC_samples(opt, burn_in=burn_in_fraction)

    return final_samples_obj.allsamples


# --- 测试块 ---
if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def target_log_pdf(x):
        mu = np.array([2, 3])
        cov = np.array([[1, 0.5], [0.5, 2]])
        inv_cov = np.linalg.inv(cov)
        delta = x - mu
        return -0.5 * np.dot(delta, np.dot(inv_cov, delta))


    n_chains = 200
    iterations_per_stage = 100
    lower_bounds = [-10, -10]
    upper_bounds = [10, 10]

    print("--- Starting SMC Sampler Test ---")
    samples_history = smc_sampler(
        log_posterior=target_log_pdf,
        n_chains=n_chains,
        iterations=iterations_per_stage,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        burn_in_fraction=0.2
    )
    print("--- SMC Sampler Test Finished ---")

    print(f"\n--- SMC 采样结果验证 ---")
    print(f"输出样本形状: {samples_history.shape}")

    final_samples = samples_history[:, -1, :]
    print(f"最终粒子位置形状: {final_samples.shape}")

    sample_mean = np.mean(final_samples, axis=0)
    sample_cov = np.cov(final_samples, rowvar=False)

    print(f"目标均值: [2.0, 3.0]")
    print(f"样本均值: [{sample_mean[0]:.4f}, {sample_mean[1]:.4f}]")
    print("\n目标协方差:\n[[1.0, 0.5]\n [0.5, 2.0]]")
    print(f"\n样本协方差:\n{sample_cov}")

    plt.figure(figsize=(8, 8))
    plt.scatter(final_samples[:, 0], final_samples[:, 1], alpha=0.5, label='SMC Final Samples')
    plt.axvline(2, color='r', linestyle='--', label='True Mean (x)')
    plt.axhline(3, color='g', linestyle='--', label='True Mean (y)')
    plt.title("Final Samples from SMC Sampler")
    plt.xlabel("Parameter 1")
    plt.ylabel("Parameter 2")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
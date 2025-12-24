#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include "DataGenerator.hpp"
#include "ParallelTempering.hpp"
#include "SimulatedAnnealing.hpp"
#include "SMC.hpp"

// =========================================================
// 全局设定：N=75, P=150
// =========================================================

void p3()
{
    const int N_SAMPLES = 75;
    const int N_FEATURES = 150;
    const int N_ACTIVE = 10;

    struct GrandBest
    {
        double f1;
        double data_s;
        double lambda;
        double sigma;
        std::vector<int> pred_indices;
        Eigen::VectorXd beta_est;
        std::vector<int> true_indices;
        std::vector<double> true_beta_vec;
    };

    // 准备总记录文件
    std::ofstream csv("logs/experiment_full_noise_sweep.csv");
    csv << "true_noise_s,lambda,model_sigma,precision,recall,f1_score,num_selected\n";

    std::cout << "================================================================================\n";
    std::cout << "  GLOBAL OPTIMUM SEARCH: Multi-Noise & Multi-Parameter Sweep\n";
    std::cout << "================================================================================\n";

    // 1. 扫描范围设置
    std::vector<double> data_noise_levels = {0.5, 1.0, 2.0, 3.0};

    std::vector<double> lambdas;
    lambdas.push_back(0.1);
    lambdas.push_back(1.0);
    lambdas.push_back(5.0);
    lambdas.push_back(10.0);
    // 黄金区加密搜索
    for (double l = 15.0; l <= 120; l += 2.5)
        lambdas.push_back(l);

    std::vector<double> model_sigmas = {0.5, 0.8, 1.0, 1.2, 2.0};

    // 初始化全局最优记录器
    GrandBest global_best = {-1.0, 0, 0, 0, {}, {}, {}, {}};

    // --- 最外层循环：数据真实噪声 ---
    for (double true_s : data_noise_levels)
    {

        std::cout << "\n\n>>> [Data Context] Generating New Data with Noise s = " << true_s << "...\n";

        // 生成当前噪声水平下的固定数据
        DataGenerator dataGen(N_SAMPLES, N_FEATURES, N_ACTIVE, true_s);
        dataGen.generate();

        auto X_std = dataGen.getX();
        auto y_std = dataGen.getY();
        Eigen::MatrixXd X(N_SAMPLES, N_FEATURES);
        Eigen::VectorXd y(N_SAMPLES);
        for (int i = 0; i < N_SAMPLES; ++i)
        {
            y(i) = y_std[i];
            for (int j = 0; j < N_FEATURES; ++j)
                X(i, j) = X_std[i * N_FEATURES + j];
        }

        // --- 恢复你喜欢的表格头 ---
        std::cout << "    Scanning Parameters (Lambda/Sigma)...\n";
        std::cout << "    -----------------------------------------------------------------------\n";
        std::cout << "    Lambda | ModSigma |  Prec  |  Rec   |   F1   | Size | Status\n";
        std::cout << "    -----------------------------------------------------------------------\n";

        // --- 参数扫描 ---
        for (double mod_sig : model_sigmas)
        {
            for (double lam : lambdas)
            {

                // 配置 PT
                ParallelTempering::Config conf;
                conf.lambda = lam;
                conf.sigma_sq = mod_sig * mod_sig;
                conf.n_chains = 36;
                conf.T_min = 1.0;
                conf.T_max = 50.0;
                conf.swap_interval = 10;

                // 动态迭代策略：核心区跑久一点，边缘区跑快一点
                if (lam >= 20 && lam <= 100)
                    conf.iterations = 3000;
                else
                    conf.iterations = 3000;

                ParallelTempering pt(X, y, conf);
                std::vector<int> pred_idx = pt.run("logs/tmp"); // 临时运行，不覆盖 VIP

                // --- 计算指标 ---
                int tp = 0, fp = 0;
                const auto &truth = dataGen.getTrueSupport();
                for (int pred : pred_idx)
                {
                    bool found = false;
                    for (int t : truth)
                        if (t == pred)
                            found = true;
                    if (found)
                        tp++;
                    else
                        fp++;
                }
                double p = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
                double r = (truth.size() > 0) ? (double)tp / truth.size() : 0.0;
                double f1 = (p + r > 0) ? 2 * p * r / (p + r) : 0.0;

                // 写入汇总 CSV
                csv << true_s << "," << lam << "," << mod_sig << "," << p << "," << r << "," << f1 << "," << pred_idx.size() << "\n";

                // --- 恢复控制台输出逻辑 ---
                if (f1 > 0.01)
                {
                    std::cout << "    " << std::setw(6) << lam << " | "
                              << std::setw(8) << mod_sig << " | "
                              << std::fixed << std::setprecision(2) << p << " | "
                              << std::setw(6) << r << " | "
                              << std::setw(6) << f1 << " | "
                              << std::setw(4) << pred_idx.size();

                    // 添加状态标签
                    if (f1 > global_best.f1)
                        std::cout << " [NEW GLOBAL BEST!]"; // 标记破纪录
                    else if (f1 > 0.85)
                        std::cout << " (High)";

                    std::cout << "\n";
                }

                // --- 捕捉全局最优并保存 VIP ---
                if (f1 > global_best.f1)
                {
                    global_best = {f1, true_s, lam, mod_sig, pred_idx, pt.getBestBeta(), truth, dataGen.getTrueBeta()};

                    // 立即保存 VIp,防止被后续循环覆盖
                    pt.run("logs/global_best_model");

                    // 【立即保存真值】
                    std::ofstream truth_file("logs/global_best_truth.csv");
                    truth_file << "true_index\n";
                    for (int idx : truth)
                        truth_file << idx << "\n";
                    truth_file.close();
                }
            }
        }
    }
    csv.close();

    std::cout << "\n================================================================================\n";
    std::cout << "  GLOBAL OPTIMUM SUMMARY\n";
    std::cout << "================================================================================\n";
    std::cout << "Highest F1-Score: " << global_best.f1 << "\n";
    std::cout << "Achieved at:      s=" << global_best.data_s << ", Lambda=" << global_best.lambda << ", Sigma=" << global_best.sigma << "\n";
    std::cout << "Selected Vars:    " << global_best.pred_indices.size() << " (Truth: " << N_ACTIVE << ")\n";
    std::cout << "\n[Files Generated]\n";
    std::cout << "1. Summary Log:   logs/experiment_full_noise_sweep.csv\n";
    std::cout << "2. Best VIP Data: logs/global_best_model_vip.csv\n";
    std::cout << "3. Best Truth:    logs/global_best_truth.csv\n";
}

// =========================================================
// 核心修复：将 0/1 向量转换为索引列表
// =========================================================
std::vector<int> mask_to_indices(const std::vector<int>& z) {
    std::vector<int> indices;
    for (size_t i = 0; i < z.size(); ++i) {
        if (z[i] == 1) indices.push_back(i);
    }
    return indices;
}

// 辅助函数：打印索引向量
void print_indices(const std::string& label, std::vector<int> indices) {
    std::sort(indices.begin(), indices.end());
    std::cout << "    " << label << " [" << indices.size() << "]: { ";
    if (indices.empty()) {
        std::cout << "(None)";
    } else {
        // 为了防止刷屏，如果超过 20 个只打印前 20 个
        int limit = 20;
        for (size_t i = 0; i < indices.size() && i < limit; ++i) {
            std::cout << indices[i] << (i < indices.size() - 1 || i == limit - 1 ? ", " : "");
        }
        if (indices.size() > limit) std::cout << "...";
    }
    std::cout << " }" << std::endl;
}

// 辅助函数：计算详细指标 (传入的必须是 Index List)
double calculate_metrics(const std::vector<int>& pred_indices, const std::vector<int>& truth, double& precision, double& recall) {
    if (pred_indices.empty()) { precision = 0.0; recall = 0.0; return 0.0; }
    int tp = 0;
    for (int p : pred_indices) {
        for (int t : truth) { if (p == t) { tp++; break; } }
    }
    precision = (double)tp / pred_indices.size();
    recall = (truth.size() > 0) ? (double)tp / truth.size() : 0.0;
    return (precision + recall > 0) ? (2.0 * precision * recall) / (precision + recall) : 0.0;
}

void save_trace(const std::string& filename, const std::vector<double>& trace, double time_sec) {
    std::ofstream f(filename);
    f << "iter,energy,time_total\n";
    for(size_t i=0; i<trace.size(); ++i) {
        f << i << "," << trace[i] << "," << time_sec << "\n";
    }
    f.close();
}

void save_ess(const std::string& filename, const std::vector<double>& ess) {
    std::ofstream f(filename);
    f << "step,ess\n";
    for(size_t i=0; i<ess.size(); ++i) {
        f << i << "," << ess[i] << "\n";
    }
    f.close();
}

void q4()
{
    // 1. 生成固定数据
    int N=60, P=150, K=10;
    double s_noise = 3.5; 
    DataGenerator gen(N, P, K, s_noise);
    gen.generate();
    
    auto X_vec = gen.getX();
    auto y_vec = gen.getY();
    Eigen::MatrixXd X(N, P);
    Eigen::VectorXd y(N);
    for(int i=0; i<N; ++i) y(i) = y_vec[i];
    for(int i=0; i<N; ++i) for(int j=0; j<P; ++j) X(i,j) = X_vec[i*P+j];

    const std::vector<int> ground_truth = gen.getTrueSupport();


    double lambda = 13.0; 
    double sigma_sq = 1.0;

    std::cout << "==========================================================\n";
    std::cout << "  PROBLEM 4: SA vs SMC PERFORMANCE BENCHMARK\n";
    std::cout << "==========================================================\n";
    
    print_indices("Ground Truth", ground_truth);
    std::cout << "----------------------------------------------------------\n";

    // 结果变量
    std::vector<double> sa_trace;
    double sa_duration = 0.0, sa_f1 = 0.0, sa_final_energy = 0.0;
    double sa_prec = 0.0, sa_rec = 0.0;
    std::vector<int> sa_best_z_mask; // 存放 0/1 向量
    std::vector<int> sa_best_indices; // 存放索引

    std::vector<double> smc_energy_trace, smc_ess_trace;
    double smc_duration = 0.0, smc_f1 = 0.0, smc_final_energy = 0.0;
    double smc_prec = 0.0, smc_rec = 0.0;
    std::vector<int> smc_best_z_mask; // 存放 0/1 向量
    std::vector<int> smc_best_indices; // 存放索引

    // ==========================================
    // 2. 运行模拟退火 (Simulated Annealing)
    // ==========================================
    {
        std::cout << ">>> [1/2] Running Simulated Annealing (SA)..." << std::endl;
        SimulatedAnnealing::Config conf;
        conf.lambda = lambda;
        conf.sigma_sq = sigma_sq;
        conf.T_start = 100.0;
        conf.T_end = 1; // 降温更彻底一点
        conf.iterations = 200000; 

        auto start = std::chrono::high_resolution_clock::now();
        SimulatedAnnealing sa(X, y, conf);
        sa_trace = sa.run(sa_best_z_mask); // 获取 Mask
        auto end = std::chrono::high_resolution_clock::now();
        sa_duration = std::chrono::duration<double>(end-start).count();
        
        // 【关键修复】转换 Mask -> Indices
        sa_best_indices = mask_to_indices(sa_best_z_mask);
        
        // 计算详细指标
        sa_f1 = calculate_metrics(sa_best_indices, ground_truth, sa_prec, sa_rec);
        sa_final_energy = sa_trace.back();

        std::cout << "    [Done] Time: " << std::fixed << std::setprecision(4) << sa_duration << "s" << std::endl;
        std::cout << "    Metrics: F1=" << std::setprecision(2) << sa_f1 
                  << " (Prec=" << sa_prec << ", Rec=" << sa_rec << ")" << std::endl;
        std::cout << "    Energy : " << std::setprecision(4) << sa_final_energy << std::endl;
        print_indices("Selected", sa_best_indices); // 打印索引
        
        save_trace("logs/q4_sa_trace.csv", sa_trace, sa_duration);
    }

    std::cout << "----------------------------------------------------------\n";

    // ==========================================
    // 3. 运行序列蒙特卡洛 (SMC)
    // ==========================================
    {
        std::cout << ">>> [2/2] Running Sequential Monte Carlo (SMC)..." << std::endl;
        SMC::Config conf;
        conf.lambda = lambda;
        conf.sigma_sq = sigma_sq;
        conf.n_particles = 3000; 
        conf.steps = 100;
        conf.mcmc_steps = 2; // 稍微增加一点内部变异

        auto start = std::chrono::high_resolution_clock::now();
        SMC smc(X, y, conf);
        smc.run(smc_best_z_mask, smc_energy_trace, smc_ess_trace);
        auto end = std::chrono::high_resolution_clock::now();
        smc_duration = std::chrono::duration<double>(end-start).count();

        // 【关键修复】转换 Mask -> Indices
        smc_best_indices = mask_to_indices(smc_best_z_mask);

        // 计算详细指标
        smc_f1 = calculate_metrics(smc_best_indices, ground_truth, smc_prec, smc_rec);
        smc_final_energy = smc_energy_trace.back();

        std::cout << "    [Done] Time: " << std::fixed << std::setprecision(4) << smc_duration << "s" << std::endl;
        std::cout << "    Metrics: F1=" << std::setprecision(2) << smc_f1 
                  << " (Prec=" << smc_prec << ", Rec=" << smc_rec << ")" << std::endl;
        std::cout << "    Energy : " << std::setprecision(4) << smc_final_energy << std::endl;
        print_indices("Selected", smc_best_indices); // 打印索引

        save_trace("logs/q4_smc_trace.csv", smc_energy_trace, smc_duration);
        save_ess("logs/q4_smc_ess.csv", smc_ess_trace);
    }

    // ==========================================
    // 4. 保存指标 & 打印最终对比表
    // ==========================================
    std::ofstream metric_file("logs/q4_metrics.csv");
    metric_file << "Metric,SA,SMC\n";
    metric_file << "Energy," << sa_final_energy << "," << smc_final_energy << "\n";
    metric_file << "Time_Seconds," << sa_duration << "," << smc_duration << "\n";
    metric_file << "F1_Score," << sa_f1 << "," << smc_f1 << "\n";
    metric_file.close();

    double speedup = (smc_duration > 0) ? sa_duration / smc_duration : 0.0;
    std::cout << "==========================================================\n";
    std::cout << "  FINAL COMPARISON SUMMARY\n";
    std::cout << "==========================================================\n";
    std::cout << "Metric         | SA (Serial)    | SMC (Parallel)\n";
    std::cout << "---------------|----------------|----------------\n";
    std::cout << "Time (s)       | " << std::setw(14) << sa_duration << " | " << std::setw(14) << smc_duration << "\n";
    std::cout << "Energy (Min)   | " << std::setw(14) << sa_final_energy << " | " << std::setw(14) << smc_final_energy << "\n";
    std::cout << "F1 Score       | " << std::setw(14) << sa_f1 << " | " << std::setw(14) << smc_f1 << "\n";
    std::cout << "Var Count      | " << std::setw(14) << sa_best_indices.size() << " | " << std::setw(14) << smc_best_indices.size() << "\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << ">>> Speedup: SMC is " << std::fixed << std::setprecision(1) << speedup << "x faster than SA.\n";
    std::cout << ">>> Metrics saved to logs/q4_metrics.csv" << std::endl;
}

int main()
{
    q4();
    return 0;
}
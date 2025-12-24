#include "ParallelTempering.hpp"
#include "LSSolver.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <omp.h> // 必须包含 OpenMP 头文件

// 构造函数
ParallelTempering::ParallelTempering(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, Config config)
    : X_full(X), y_full(y), conf(config) {
    n_samples = X.rows();
    n_features = X.cols();
    global_best_energy = 1e30; 
}

// 计算能量 (Energy = RSS/2s^2 + lambda*k)
double ParallelTempering::compute_energy(const std::vector<int>& z, double& out_rss) {
    std::vector<int> active_indices;
    active_indices.reserve(n_features); // 预分配内存优化
    for (int i = 0; i < n_features; ++i) {
        if (z[i] == 1) active_indices.push_back(i);
    }
    
    int k = active_indices.size();
    
    // 如果没有选中任何变量
    if (k == 0) {
        out_rss = y_full.squaredNorm();
        return out_rss / (2.0 * conf.sigma_sq);
    }

    // 构造子矩阵 (这里会频繁调用，是性能瓶颈，Eigen 操作要快)
    Eigen::MatrixXd X_sub(n_samples, k);
    for (int i = 0; i < k; ++i) {
        X_sub.col(i) = X_full.col(active_indices[i]);
    }

    // 调用 Solver
    auto res = LSSolver::solve(X_sub, y_full);
    out_rss = res.rss;
    
    // 贝叶斯 BIC 风格能量函数
    return (out_rss / (2.0 * conf.sigma_sq)) + (conf.lambda * k);
}

// 改进后的提议分布 (Propose Move)
// 引入混合翻转策略：既能微调(1 bit)，也能跳跃(3 bits)，帮助逃离局部最优
void ParallelTempering::propose_move(std::vector<int>& z, std::mt19937& gen) {
    std::uniform_real_distribution<double> u(0, 1);
    std::uniform_int_distribution<int> d(0, n_features - 1);
    
    // 85% 概率翻转 1 个变量 (精细搜索)
    // 15% 概率翻转 2-3 个变量 (大步跳跃，更容易抓到被掩盖的弱信号)
    int flips = (u(gen) < 0.85) ? 1 : (d(gen) % 2 + 2); 

    for(int f = 0; f < flips; ++f) {
        int idx = d(gen);
        z[idx] = 1 - z[idx]; // 0变1，1变0
    }
}

std::vector<int> ParallelTempering::run(const std::string& log_prefix) {
    // 1. 初始化温度阶梯
    std::vector<double> temperatures(conf.n_chains);
    double steps = std::max(1, conf.n_chains - 1);
    double factor = std::pow(conf.T_max / conf.T_min, 1.0 / steps);
    for (int i = 0; i < conf.n_chains; ++i) {
        temperatures[i] = conf.T_min * std::pow(factor, i);
    }

    // 2. 初始化链
    std::vector<ChainState> chains(conf.n_chains);
    omp_set_num_threads(conf.n_chains);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < conf.n_chains; ++i) {
        std::mt19937 gen(42 + i);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        chains[i].z.resize(n_features);
        for (int j = 0; j < n_features; ++j) {
            chains[i].z[j] = (dist(gen) < 0.05) ? 1 : 0;
        }
        chains[i].energy = compute_energy(chains[i].z, chains[i].rss);
    }

    // 3. 统计变量 (用于 VIP)
    std::vector<long long> selection_counts(n_features, 0); // 用 long long 防止溢出
    long long valid_samples = 0;

    std::mt19937 swap_gen(9999);
    std::uniform_real_distribution<double> swap_uni(0.0, 1.0);

    // ==========================================
    // MCMC 主循环
    // ==========================================
    for (int iter = 0; iter < conf.iterations; ++iter) {
        
        // A. 并行 Metropolis 步
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < conf.n_chains; ++i) {
            std::mt19937 gen(12345 + iter * 7 + i * 999);
            std::uniform_real_distribution<double> uni(0.0, 1.0);

            for(int s = 0; s < conf.swap_interval; ++s) {
                std::vector<int> cand_z = chains[i].z;
                propose_move(cand_z, gen);

                double cand_rss;
                double cand_E = compute_energy(cand_z, cand_rss);
                double delta = cand_E - chains[i].energy;

                if (delta < 0 || uni(gen) < std::exp(-delta / temperatures[i])) {
                    chains[i].z = cand_z;
                    chains[i].energy = cand_E;
                    chains[i].rss = cand_rss;
                }
            }
        } 

        // B. 链间交换
        for (int i = conf.n_chains - 1; i > 0; --i) {
            double log_p = (chains[i-1].energy - chains[i].energy) * (1.0/temperatures[i] - 1.0/temperatures[i-1]);
            if (log_p > 0 || swap_uni(swap_gen) < std::exp(log_p)) {
                std::swap(chains[i], chains[i-1]);
            }
        }

        // C. 更新全局最优
        if (chains[0].energy < global_best_energy) {
            global_best_energy = chains[0].energy;
            global_best_z = chains[0].z;
        }

        // D. 统计 VIP (只在 Burn-in 之后统计)
        if (iter > conf.iterations * 0.2) {
            valid_samples++;
            for(int j = 0; j < n_features; ++j) {
                if(chains[0].z[j] == 1) selection_counts[j]++;
            }
        }
    }

    // ==========================================
    // 4.写入 VIP CSV 文件
    // ==========================================
    std::string vip_filename = log_prefix + "_vip.csv";
    std::ofstream vip_file(vip_filename);
    
    if (vip_file.is_open()) {
        vip_file << "feature_index,probability\n";
        for (int j = 0; j < n_features; ++j) {
            double prob = (valid_samples > 0) ? (double)selection_counts[j] / valid_samples : 0.0;
            vip_file << j << "," << prob << "\n";
        }
        vip_file.close();
        // std::cout << "Saved VIP data to " << vip_filename << std::endl;
    } else {
        std::cerr << "Error: Could not open " << vip_filename << " for writing." << std::endl;
    }

    // 返回全局最优解
    std::vector<int> result;
    for(int i = 0; i < n_features; ++i) {
        if(global_best_z[i] == 1) result.push_back(i);
    }
    return result;
}

// 获取系数
Eigen::VectorXd ParallelTempering::getBestBeta() const {
    std::vector<int> idx;
    for(int i=0; i<n_features; ++i) if(global_best_z[i]) idx.push_back(i);
    
    if (idx.empty()) return Eigen::VectorXd::Zero(n_features);

    Eigen::MatrixXd X_sub(n_samples, idx.size());
    for(size_t k=0; k<idx.size(); ++k) X_sub.col(k) = X_full.col(idx[k]);
    
    auto res = LSSolver::solve(X_sub, y_full);
    
    Eigen::VectorXd full_beta = Eigen::VectorXd::Zero(n_features);
    for(size_t k=0; k<idx.size(); ++k) full_beta(idx[k]) = res.beta(k);
    
    return full_beta;
}
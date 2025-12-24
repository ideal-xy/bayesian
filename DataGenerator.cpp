// src/DataGenerator.cpp
#include "DataGenerator.hpp"
#include <iostream>
#include <fstream> // 文件流
#include <iomanip> // 设置精度
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>

DataGenerator::DataGenerator(int n, int d, int m, double sigma)
    : n_samples(n), n_features(d), n_active(m), noise_sigma(sigma)
{
    X.resize(n * d);
    y.resize(n);
    true_beta.resize(d, 0.0);
}

void DataGenerator::generate()
{
    std::mt19937 gen(42); // 固定种子42，保证结果可复现

    // 1. 生成 X
    std::normal_distribution<double> dist_X(0.0, 1.0);
    for (int i = 0; i < n_samples * n_features; ++i)
    {
        X[i] = dist_X(gen);
    }

    // 2. 随机选取重要变量
    std::vector<int> indices(n_features);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    true_support.clear();
    for (int i = 0; i < n_active; ++i)
    {
        true_support.push_back(indices[i]);
    }
    std::sort(true_support.begin(), true_support.end());

    // 3. 生成 Beta
    std::uniform_real_distribution<double> dist_beta(-5.0, 5.0);
    for (int idx : true_support)
    {
        true_beta[idx] = dist_beta(gen);
    }

    // 4. 生成 Y (OpenMP 加速)
    std::normal_distribution<double> dist_eps(0.0, noise_sigma);

    // 预生成噪声保持单线程一致性
    std::vector<double> noise(n_samples);
    for (int i = 0; i < n_samples; ++i)
        noise[i] = dist_eps(gen);

#pragma omp parallel for
    for (int i = 0; i < n_samples; ++i)
    {
        double dot_product = 0.0;
        for (int j = 0; j < n_features; ++j)
        {
            if (std::abs(true_beta[j]) > 1e-9)
            {
                dot_product += X[i * n_features + j] * true_beta[j];
            }
        }
        y[i] = dot_product + noise[i];
    }
}

void DataGenerator::printInfo() const
{
    std::cout << "=== Data Generation Info ===\n";
    std::cout << "Samples: " << n_samples << ", Features: " << n_features << "\n";
    std::cout << "True Active Indices: ";
    for (int idx : true_support)
        std::cout << idx << " ";
    std::cout << "\n============================\n";
}

void DataGenerator::saveToFile(const std::string &filename) const
{
    // 1. 保存观测数据 (Solver 读取用)
    std::ofstream out(filename);
    if (!out.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    // 格式：
    // 第一行：样本数N 特征数D
    // 后续N行：y x_0 x_1 ... x_d-1
    out << n_samples << " " << n_features << "\n";
    out << std::fixed << std::setprecision(6); // 保留6位小数

    for (int i = 0; i < n_samples; ++i)
    {
        out << y[i]; // 先写 y
        for (int j = 0; j < n_features; ++j)
        {
            out << " " << X[i * n_features + j]; // 再写一行 x
        }
        out << "\n";
    }
    out.close();
    std::cout << "[Saved] Observation data saved to: " << filename << "\n";

    // 2. 保存真值 (Verification 核对答案用)
    std::string truthFile = filename + ".truth";
    std::ofstream outTrue(truthFile);
    if (!outTrue.is_open())
        return;

    // 格式：
    // 第一行：真实变量个数 M
    // 后续 M 行：变量下标 真实系数Beta值
    outTrue << n_active << "\n";
    outTrue << std::fixed << std::setprecision(6);

    for (int idx : true_support)
    {
        outTrue << idx << " " << true_beta[idx] << "\n";
    }
    outTrue.close();
    std::cout << "[Saved] Ground truth saved to: " << truthFile << "\n";
}
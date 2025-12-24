#include "LSSolver.hpp"

LSSolution LSSolver::solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr;
    qr.compute(A);
    
    // 求解线性方程组 R*x = Q^T * b
    Eigen::VectorXd beta = qr.solve(b);
    
    // 计算残差平方和 RSS = ||y - X*beta||^2
    double rss = (A * beta - b).squaredNorm();
    
    return {beta, rss};
}
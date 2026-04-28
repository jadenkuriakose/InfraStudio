#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

extern "C" {

void computeMetrics(
    const double* predictions,
    const double* labels,
    int n,
    double* outAccuracy,
    double* outF1
) {
    double correct = 0.0;
    double tp = 0.0, fp = 0.0, fn = 0.0;

    for (int i = 0; i < n; ++i) {
        int pred = static_cast<int>(std::round(predictions[i]));
        int label = static_cast<int>(std::round(labels[i]));
        if (pred == label) correct += 1.0;
        if (pred == 1 && label == 1) tp += 1.0;
        if (pred == 1 && label == 0) fp += 1.0;
        if (pred == 0 && label == 1) fn += 1.0;
    }

    *outAccuracy = (n > 0) ? correct / n : 0.0;

    double precision = (tp + fp > 0) ? tp / (tp + fp) : 0.0;
    double recall    = (tp + fn > 0) ? tp / (tp + fn) : 0.0;
    *outF1 = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
}

void rollingStats(
    const double* values,
    int n,
    int windowSize,
    double* outMeans,
    double* outStds
) {
    for (int i = 0; i <= n - windowSize; ++i) {
        double sum = 0.0;
        for (int j = i; j < i + windowSize; ++j) sum += values[j];
        double mean = sum / windowSize;
        double variance = 0.0;
        for (int j = i; j < i + windowSize; ++j) {
            double diff = values[j] - mean;
            variance += diff * diff;
        }
        outMeans[i] = mean;
        outStds[i]  = std::sqrt(variance / windowSize);
    }
}

double sharpeRatio(const double* returns, int n, double riskFreeRate) {
    if (n < 2) return 0.0;
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += returns[i];
    mean /= n;
    double variance = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = returns[i] - mean;
        variance += diff * diff;
    }
    double stdDev = std::sqrt(variance / (n - 1));
    return (stdDev > 0) ? (mean - riskFreeRate) / stdDev : 0.0;
}

double maxDrawdown(const double* equityCurve, int n) {
    double peak = equityCurve[0];
    double maxDD = 0.0;
    for (int i = 1; i < n; ++i) {
        if (equityCurve[i] > peak) peak = equityCurve[i];
        double dd = (peak - equityCurve[i]) / peak;
        if (dd > maxDD) maxDD = dd;
    }
    return maxDD;
}

} // extern "C"
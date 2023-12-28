#include <vector>
#include <cmath>
#include <assert.h>

using namespace std;

class Layer {
    public:
        int n, prev_n;
        vector<vector<double>> w;
        vector<double> b;

    public:
        Layer() {}
        Layer(int N, int prev_N) {
            n = N;
            prev_n = prev_N;
            w = vector<vector<double>>(prev_n, vector<double>(n));
            for(int i = 0; i < prev_n; i++) {
                for(int j = 0; j < n; j++) {
                    w[i][j] = (double) rand() / RAND_MAX;
                    w[i][j] *= 2; w[i][j] -= 1;
                }
            }
            b = vector<double>(n);
            for(int i = 0; i < n; i++) {
                b[i] = (double) rand() / RAND_MAX;
                b[i] *= 2; b[i] -= 1;
            }
        }

        Layer(int N, int prev_N, const vector<vector<double>> &W, const vector<double> &B) {
            n = N;
            prev_n = prev_N;
            w = W;
            b = B;
        }

        vector<double> process(const vector<double> &a);
};

double sigmoid(long double x) {
    return 1 / (1 + exp((long double) -x));
}

double sigmoid_derivative(long double x) {
    double ans = sigmoid(x);
    return ans * (1 - ans);
}

vector<double> Layer::process(const vector<double> &a) {
    assert((int) a.size() == prev_n);
    vector<double> ans = b;
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < prev_n; i++) {
            ans[j] += w[i][j] * a[i];
        }
    }
    return ans;
}

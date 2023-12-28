#include "entry.h"
#include <string>
#include <iomanip>

using namespace std;

class network {
    public:
        int layer_count;
        vector<int> layer_sizes;
        vector<Layer> layers;
        string path;

    public:
    network() {}
    network(const vector<int> &sizes) {
        layer_sizes = sizes;
        layer_count = sizes.size();
        layers = vector<Layer>(layer_count);
        for(int i = 1; i < layer_count; i++) {
            layers[i] = Layer(sizes[i], sizes[i - 1]);
        }
    }

    vector<double> run(vector<double> a) {
        for(int i = 1; i < layer_count; i++) {
            a = layers[i].process(a);
            for(auto &x : a) x = sigmoid(x);
        }
//        for(auto &x : a) cout << fixed << setprecision(5) << x << ' '; cout << endl;
        return a;
    }

    int make_guess(const entry &e) {
        vector<double> a = e.input;
        a = run(a);
        int answer = -1;
        double maximum = -1;
        for(int i = 0; i < 10; i++) {
            if(maximum < a[i]) {
                maximum = a[i];
                answer = i;
            }
        }
        assert(~answer);
        return answer;
    }

    double get_error(entry &e) {
        vector<double> a = e.input;
        double err = e.evaluate(run(a));
        return err;
    }

    vector<double> get_probability(entry &e) {
        vector<double> a = e.input;
        a = run(a);
        double sum = 0;
        for(auto &p : a) sum += p;
        for(auto &p : a) p /= sum;
        return a;
    }

    pair<vector<vector<vector<double>>>, vector<vector<double>>> get_gradient(entry e) {
        vector<vector<double>> unsigmoided(layer_count);
        vector<vector<double>> sigmoided(layer_count);
        unsigmoided[0] = sigmoided[0] = e.input;
        for(int i = 1; i < layer_count; i++) {
            unsigmoided[i] = layers[i].process(sigmoided[i - 1]);
            sigmoided[i] = unsigmoided[i];
            for(auto &x : sigmoided[i]) x = sigmoid(x);
        }
        vector<double> partial_derivatives(layers.back().n);
        for(int i = 0; i < layers.back().n; i++) {
            partial_derivatives[i] = 2 * (sigmoided.back()[i] - e.output[i]) * sigmoid_derivative(unsigmoided.back()[i]);
        }
        vector<vector<vector<double>>> gradients_weights(layer_count);
        vector<vector<double>> gradients_baises(layer_count);
        for(int i = layer_count - 1; i; i--) {
            gradients_weights[i] = vector<vector<double>>(layers[i].prev_n, vector<double>(layers[i].n));
            gradients_baises[i] = vector<double>(layers[i].n);
            for(int v = 0; v < layers[i].prev_n; v++) {
                for(int u = 0; u < layers[i].n; u++) {
                    gradients_weights[i][v][u] = sigmoided[i - 1][v] * partial_derivatives[u];
                }
            }
            for(int u = 0; u < layers[i].n; u++) {
                gradients_baises[i][u] = partial_derivatives[u];
            }
            vector<double> new_partial_derivatives(layers[i].prev_n);
            for(int v = 0; v < layers[i].prev_n; v++) {
                for(int u = 0; u < layers[i].n; u++) {
                    new_partial_derivatives[v] += layers[i].w[v][u] * partial_derivatives[u];
                }
                new_partial_derivatives[v] *= sigmoid_derivative(unsigmoided[i - 1][v]);
            }
            partial_derivatives = new_partial_derivatives;
        }
        return make_pair(gradients_weights, gradients_baises);
    }

    void adjust_weights(vector<entry> data, const double learn_rate = 3) {
        vector<vector<vector<double>>> gradients_weights(layer_count);
        vector<vector<double>> gradients_baises(layer_count);
        for(int i = 1; i < layer_count; i++) {
            gradients_weights[i] = vector<vector<double>>(layers[i].prev_n, vector<double>(layers[i].n));
            gradients_baises[i] = vector<double>(layers[i].n);
        }
        for(entry &e : data) {
            auto [curr_gradients_weights, curr_gradients_baises] = get_gradient(e);
            for(int i = 1; i < layer_count; i++) {
                for(int v = 0; v < layers[i].prev_n; v++) {
                    for(int u = 0; u < layers[i].n; u++) {
                        gradients_weights[i][v][u] += curr_gradients_weights[i][v][u];
                    }
                }
                for(int u = 0; u < layers[i].n; u++) {
                    gradients_baises[i][u] += curr_gradients_baises[i][u];
                }
            }
        }
        for(int i = 1; i < layer_count; i++) {
            for(int v = 0; v < layers[i].prev_n; v++) {
                for(int u = 0; u < layers[i].n; u++) {
                    gradients_weights[i][v][u] /= (double) data.size();
                }
            }
            for(int u = 0; u < layers[i].n; u++) {
                gradients_baises[i][u] /= (double) data.size();
            }
        }
        for(int i = 1; i < layer_count; i++) {
            for(int v = 0; v < layers[i].prev_n; v++) {
                for(int u = 0; u < layers[i].n; u++) {
                    layers[i].w[v][u] -= gradients_weights[i][v][u] * learn_rate;
                }
            }
            for(int u = 0; u < layers[i].n; u++) {
                layers[i].b[u] -= gradients_baises[i][u] * learn_rate;
            }
        }
    }

    pair<double, double> test(const vector<entry> &data) {
        int cnt = 0;
        double err = 0;
        int it = 0;
        for(auto e : data) {
            err += get_error(e);
            cnt += (make_guess(e) == e.answer);
            it++;
        }
        pair<double, double> ans = {(double) cnt / data.size(), err / data.size()};
        return ans;
    }

    void train(const vector<entry> &data, int iterations, double learn_rate) {
        const int group_size = 100;
        vector<vector<entry>> groups;
        for(int i = 0; i < (int) data.size(); i += group_size) {
            vector<entry> group;
            for(int j = i; j < (int) data.size() && j < i + group_size; j++) {
                group.push_back(data[j]);
            }
            groups.push_back(group);
        }
        for(int it = 1; it <= iterations; it++) {
            for(auto &group : groups) {
                adjust_weights(group, learn_rate);
            }
            cout << "FINISHED ITERATION " << it << endl;
            auto [accuracy, penalty] = test(data);
            cout << "ACCURACY: " << accuracy << ", PENALTY: " << penalty << endl;
            ofstream fout("weights.txt");
            fout << layer_count << '\n';
            for(auto i : layer_sizes) fout << i << " "; fout << '\n';
            for(int i = 1; i < layer_count; i++) {
                for(int v = 0; v < layers[i].prev_n; v++) {
                    for(int u = 0; u < layers[i].n; u++) {
                        fout << layers[i].w[v][u] << '\n';
                    }
                }
                for(int u = 0; u < layers[i].n; u++) {
                    fout << layers[i].b[u] << '\n';
                }
            }
        }
    }

    void make_submission(const vector<entry> &data) {
        ofstream fout("output.csv");
        fout << "ImageId,Label" << '\n';
        int IDX = 0;
        for(auto e : data) {
            int guess = make_guess(e);
            fout << (++IDX) << "," << guess << '\n';
        }
    }
};

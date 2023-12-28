#include "layer.h"
#include <vector>
#include <assert.h>

using namespace std;

class entry {
    public:
        vector<double> input, output;
        int answer;

    public:
        entry(const vector<double> &Input, const vector<double> &Output) {
            input = Input;
            output = Output;
            for(int i = 0; i < 10; i++) {
                if(output[i] == 1) answer = i;
            }
        }

        double evaluate(const vector<double> &Output) {
            assert(output.size() == Output.size());
            double ans = 0;
            for(int i = 0; i < (int) output.size(); i++) {
                ans += (output[i] - Output[i]) * (output[i] - Output[i]);
            }
            return ans;
        }
};

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include "Graphics.h"

using namespace std;

vector<entry> read_data(const string &path) {
    ifstream fin(path);
    string s; getline(fin, s);
    vector<entry> data;
    while(getline(fin, s)) {
        vector<int> nums;
        for(int i = 0; i < (int) s.size(); i++) {
            if(s[i] == ',') continue;
            int j = i;
            int x = 0;
            while(j < (int) s.size() && s[j] != ',') x = 10 * x + s[j++] - '0'; j--;
            i = j;
            nums.push_back(x);
        }
        vector<double> output(10);
        output[nums[0]] = 1;
        vector<double> input(nums.size() - 1);
        assert(input.size() == 784);
        for(int i = 1; i < (int) nums.size(); i++) {
            input[i] = nums[i] / 255.0;
        }
        data.push_back(entry(input, output));
    }
    return data;
}

vector<entry> read_tests(const string &path) {
    ifstream fin(path);
    string s; getline(fin, s);
    vector<entry> data;
    while(getline(fin, s)) {
        vector<int> nums;
        for(int i = 0; i < (int) s.size(); i++) {
            if(s[i] == ',') continue;
            int j = i;
            int x = 0;
            while(j < (int) s.size() && s[j] != ',') x = 10 * x + s[j++] - '0'; j--;
            i = j;
            nums.push_back(x);
        }
        vector<double> output(10);
        vector<double> input(nums.size());
        for(int i = 0; i < (int) nums.size(); i++) {
            input[i] = nums[i] / 255.0;
        }
        data.push_back(entry(input, output));
    }
    return data;
}


network read_weights() {
    ifstream fin("weights.txt");
    int n;
    fin >> n;
    vector<int> sizes(n);
    for(auto &i : sizes) {
        fin >> i;
    }
    network nw(sizes);
    for(int i = 1; i < nw.layer_count; i++) {
        for(int v = 0; v < nw.layers[i].prev_n; v++) {
            for(int u = 0; u < nw.layers[i].n; u++) {
                fin >> nw.layers[i].w[v][u];
            }
        }
        for(int u = 0; u < nw.layers[i].n; u++) {
            fin >> nw.layers[i].b[u];
        }
    }
    return nw;
}

void pictures_train_data() {
    vector<entry> data = read_data("data/train.csv");
    int INDEX = 1;
    for(entry e : data) {
        sf::Image image;
        image.create(28, 28);
        for(int j = 0; j < 28; j++) {
            for(int i = 0; i < 28; i++) {
                int col = e.input[j * 28 + i] * 255;
                image.setPixel(i, j, sf::Color(col, col, col));
            }
        }
        image.saveToFile("data/train/image" + to_string(INDEX) + ".png");
        INDEX++;
    }
}

sf::Image transformImage(const sf::Image& inputImage) {
    srand(time(0)); // Seed for random number generation

    // Random scale factor between 0.8 and 1.2
    float scaleFactor = 0.8f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(1.2f - 0.8f)));

    // Random rotation angle between -25 and 25 degrees
    float rotationAngle = -25 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(25 - (-25))));

    // Random shift in x and y directions, within a range of -10 to 10 pixels
    float shiftX = -5 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(5 - (-5))));
    float shiftY = -5 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(5 - (-5))));

    // Create a texture and load the image into it
    sf::Texture texture;
    texture.loadFromImage(inputImage);

    // Create a sprite to apply transformations
    sf::Sprite sprite;
    sprite.setTexture(texture);
    sprite.setOrigin(inputImage.getSize().x / 2, inputImage.getSize().y / 2); // Set origin to the center of the image
    sprite.setScale(scaleFactor, scaleFactor);
    sprite.setRotation(rotationAngle);

    // Apply random shift
    sprite.setPosition(inputImage.getSize().x / 2 + shiftX, inputImage.getSize().y / 2 + shiftY);

    // Create a RenderTexture to draw the transformed sprite
    sf::RenderTexture renderTexture;
    renderTexture.create(inputImage.getSize().x, inputImage.getSize().y);
    renderTexture.clear(sf::Color::Black);
    renderTexture.draw(sprite);
    renderTexture.display();

    // Get the transformed image
    return renderTexture.getTexture().copyToImage();
}

void transform_train_data() {
    vector<entry> data = read_data("data/train.csv");
    ofstream fout("data/transformed_train.csv");
    fout << "USELESS_LINE" << '\n';
    int INDEX = 1;
    for(entry e : data) {
        sf::Image image;
        image.create(28, 28);
        for(int j = 0; j < 28; j++) {
            for(int i = 0; i < 28; i++) {
                int col = e.input[j * 28 + i] * 255;
                image.setPixel(i, j, sf::Color(col, col, col));
            }
        }
        image = transformImage(image);
        fout << e.answer;
        assert(image.getSize().x == 28);
        assert(image.getSize().y == 28);
        for(int j = 0; j < 28; j++) {
            for(int i = 0; i < 28; i++) {
                int x = image.getPixel(i, j).r;
                fout << "," << x;
            }
        }
        fout << '\n';
        if(INDEX % 1000 == 0) cout << "FINISHED " << INDEX << endl;
        INDEX++;
    }
}

void train(int iterations, double learn_rate, bool continue_training) {
    vector<entry> data = read_data("data/transformed_train.csv");
    network nw;
    if(continue_training) nw = read_weights();
    else {
        vector<int> sizes = {784, 30, 30, 10}; nw = network(sizes);
    }
    nw.train(data, iterations, learn_rate);
}

void draw() {
    network nw;
    nw = read_weights();
    run(nw);
}

void make_submission() {
    network nw;
    nw = read_weights();
    vector<entry> data = read_data("data/transformed_train.csv");
    vector<entry> tests = read_data("data/test.csv");
    auto [accuracy, penalty] = nw.test(data); cout << "ACCURACY: " << accuracy << ", PENALTY: " << penalty << endl;
    nw.make_submission(tests);
}

int main() {
    string s;
    while(cin >> s) {
        if(s == "help") {
            cout << "Commands:\n";
            cout << "train [iterations] [learn_rate] [continue_training (0, 1)]: starts the training. If continue_training is true, reads the weights from the file \"weights.txt\" and continues with them.\n";
            cout << "draw: allows the user to draw the digits and outputs the predictions to the console.\n";
            cout << "exit / break / quit / stop: terminates the program.\n";
            cout.flush();
        }
        else if(s == "train") {
            int iterations;
            double learn_rate;
            bool continue_training;
            cin >> iterations >> learn_rate >> continue_training;
            train(iterations, learn_rate, continue_training);
        }
        else if(s == "draw") {
            draw();
        }
        else if(s == "exit" || s == "break" || s == "quit" || s == "stop") {
            break;
        }
        else {
            cout << "Unknown command\n";
            cout.flush();
        }
    }
    return 0;
}

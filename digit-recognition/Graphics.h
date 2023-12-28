#include <SFML/Graphics.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>
#include "network.h"

sf::Image scaleDown(const sf::Image& originalImage) {
    auto sz = originalImage.getSize();
    int x = sz.x;
    int y = sz.y;
    sf::Texture texture;
    texture.loadFromImage(originalImage);
    sf::Sprite sprite;
    sprite.setTexture(texture);
    sprite.setScale((double) 28 / x, (double) 28 / y);
    sf::RenderTexture renderTexture;
    renderTexture.create(sprite.getGlobalBounds().width, sprite.getGlobalBounds().height);
    renderTexture.clear();
    renderTexture.draw(sprite);
    renderTexture.display();
    return renderTexture.getTexture().copyToImage();
}

sf::VertexArray color(sf::RectangleShape rectangle) {
    sf::VertexArray vertices(sf::Quads, 8);
    sf::Vector2f size = rectangle.getSize();
    sf::Vector2f localVertices[8] = {
        sf::Vector2f(0, 0),
        sf::Vector2f(size.x, 0),
        sf::Vector2f(size.x, size.y),
        sf::Vector2f(0, size.y),
        sf::Vector2f(0, size.y + 0),
        sf::Vector2f(size.x, size.y + 0),
        sf::Vector2f(size.x, size.y + size.y),
        sf::Vector2f(0, size.y + size.y)
    };
    const sf::Transform& transform = rectangle.getTransform();
    for (int i = 0; i < 8; ++i) {
        vertices[i].position = transform.transformPoint(localVertices[i]);
    }
    vertices[2].color = sf::Color::White;
    vertices[3].color = sf::Color::White;
    vertices[0].color = sf::Color::White;
    vertices[1].color = sf::Color::White;
    vertices[4].color = sf::Color::White;
    vertices[5].color = sf::Color::White;
    vertices[6].color = sf::Color::White;
    vertices[7].color = sf::Color::White;
    return vertices;
}

int whiteness(sf::Color color) {
    return color.r;
}

entry getEntry(const sf::Image &screen) {
    std::vector<double> input;
    std::vector<double> output(10);
    for(int j = 0; j < 28; j++) {
        for(int i = 0; i < 28; i++) {
            input.push_back(whiteness(screen.getPixel(i, j)) / 255.0);
        }
    }
    assert(input.size() == 784);
    return entry(input, output);
}

void run(network nw) {
    sf::RenderWindow window(sf::VideoMode(280, 280), "digit recognizer");
    bool held = 0;
    sf::Vector2f prev;
    std::vector<sf::VertexArray> rects;
    int frame = 0;
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }
        if(sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            sf::Vector2f pos = static_cast<sf::Vector2f>(sf::Mouse::getPosition(window));
            if(held) {
                const int height = 5;
                sf::Vector2f v = pos - prev;
                double len = sqrt(v.x * v.x + v.y * v.y);
                sf::RectangleShape line({len, height});
                double angle = atan2(v.y, v.x);
                line.rotate(angle * 180 / acos(-1));
                line.setPosition(prev);
                rects.push_back(color(line));
            }
            prev = pos;
            held = 1;
        }
        else held = 0;
        if(sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
            rects.clear();
        }
        window.clear();
        for (const auto &r : rects) {
            window.draw(r);
        }
        if(frame % 200 == 0) {
            sf::Image screen = scaleDown(window.capture());
            entry e = getEntry(screen);
            vector<double> p = nw.get_probability(e);
            vector<pair<double, int>> a(10);
            for(int i = 0; i < 10; i++) {
                a[i] = {p[i], i};
            }
            sort(a.rbegin(), a.rend());
            system("CLS");
            for(int i = 0; i < 10; i++) {
                cout << fixed << setprecision(5) << a[i].second << ": probability " << a[i].first << '\n';
            }
            cout.flush();
        }
        window.display();
        frame++;
    }
}

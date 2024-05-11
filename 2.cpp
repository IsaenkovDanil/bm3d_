#include <opencv2/opencv.hpp>
#include <iostream>

#define NN 8 // Размер блока DCT

int main() {
    // Предполагаем, что src уже заполнен данными размером 8x8
    float srcData[NN*NN];
    for (int i = 0; i < NN*NN; ++i) {
        srcData[i] = i; // Пример заполнения
    }

    // Создаем cv::Mat из одномерного массива
    cv::Mat src(NN, NN, CV_32F, srcData);

    std::cout << "Original Matrix:\n" << src << "\n\n";

    // Выполняем прямое DCT
    cv::Mat dst;
    cv::dct(src, dst);

    std::cout << "After DCT:\n" << dst << "\n\n";

    // Выполняем обратное DCT
    cv::Mat idst;
    cv::idct(dst, idst);

    std::cout << "After IDCT:\n" << idst << "\n\n";

    // Копируем результаты обратно в srcData, если необходимо
    for (int i = 0; i < NN; ++i) {
        for (int j = 0; j < NN; ++j) {
            srcData[i * NN + j] = idst.at<float>(i, j);
        }
    }

    // Теперь srcData содержит результаты после DCT и обратного DCT
    return 0;
}

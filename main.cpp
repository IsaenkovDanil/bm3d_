#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <sstream>
#include "bm3d.h"
#include "utilities.h"
#include <opencv2/opencv.hpp>
#define YUV       0
#define YCBCR     1
#define OPP       2
#define RGB       3
#define DCT       4
#define BIOR      5
#define HADAMARD  6
#define NONE      7
typedef uint8_t ImageType;
using namespace std;

double get_psnr(const std::vector<float>& img1, const std::vector<float>& img2, int pixels, float vmax)
{
    double mse = 0.0;
    for (int i = 0; i < pixels; ++i)
    {
        double diff = static_cast<double>(img1[i]) - static_cast<double>(img2[i]);
        mse += diff * diff;
    }
    mse /= pixels;

        return 10.0 * log10(static_cast<double>(vmax) * vmax / mse);

}

// c: pointer to original argc
// v: pointer to original argv
// o: option name after hyphen
// d: default value (if NULL, the option takes no argument)
const char *pick_option(int *c, char **v, const char *o, const char *d) {
  int id = d ? 1 : 0;
  for (int i = 0; i < *c - id; i++) {
    if (v[i][0] == '-' && 0 == strcmp(v[i] + 1, o)) {
      char *r = v[i + id] + 1 - id;
      for (int j = i; j < *c - id; j++)
        v[j] = v[j + id + 1];
      *c -= id + 1;
      return r;
    }
  }
  return d;
}

void saveChannelToCSV1( std::string& filename,  std::vector<float>& image, unsigned width, unsigned height, unsigned channel, unsigned numChannels) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл для записи: " << filename << std::endl;
        return;
    }

    for (unsigned i = 0; i < height; ++i) {
        for (unsigned j = 0; j < width; ++j) {
            // Доступ к элементу массива с учетом канала
            float value = image[i * width * numChannels + j * numChannels + channel];
            file << value;
            if (j < width - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Файл успешно сохранен: " << filename << std::endl;
}
FILE *openfile(const char *fname, const char *mode)
{
	FILE *f = fopen(fname, mode);
	if (NULL == f)
	{
		cout << "Failed to open: " << fname << endl;
		exit(1);
	}
	return f;
}


std::vector<float> imageToVector(const std::string& filename) {
    // Загрузка изображения в формате CV_32F (32-битные вещественные числа)
    cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Could not open or find the image\n";
        return {};
    }

    // Преобразование изображения в тип float, если это еще не сделано
    if (image.type() != CV_32F) {
        image.convertTo(image, CV_32F);
    }

    // Нормализация значений пикселей, если нужно (например, приведение к диапазону [0, 1])
    //image /= 255.0;

    // Перевод изображения в одномерный вектор
    std::vector<float> vector;
    vector.assign((float*)image.datastart, (float*)image.dataend);
    return vector;
}


/**
 * @file   main.cpp
 * @brief  Main executable file. Do not use lib_fftw to
 *         process DCT.
 *
 * @author MARC LEBRUN  <marc.lebrun@cmla.ens-cachan.fr>
 */
int main(int argc, char **argv)
{argc=12;
argv[1]="cinput.png";

       argv[2]=         "35";
     argv[3]=           "ImDenoised.png";
      argv[4]=          "ImBasic.png";
       argv[5]=        "-useSD_wien";
       argv[6]=         "-tau_2d_hard";
       argv[7]=         "bior";
        argv[8]=        "-tau_2d_wien";
       argv[9]=         "dct";
       argv[10]=         "-color_space";
        argv[11]=        "rgb";
  argv[12]=        "-nb_threads";
    argv[13]=        "1";
       argv[14]=        "-verbose";
  //! Variables initialization
  const char *_tau_2D_hard = pick_option(&argc, argv, "tau_2d_hard", "bior");
  const char *_tau_2D_wien = pick_option(&argc, argv, "tau_2d_wien", "dct");
  const char *_color_space = pick_option(&argc, argv, "color_space", "opp");
  const char *_patch_size = pick_option(&argc, argv, "patch_size", "0"); // >0: overrides default
  const char *_nb_threads = pick_option(&argc, argv, "nb_threads", "0");
  const bool useSD_1 = pick_option(&argc, argv, "useSD_hard", NULL) != NULL;
  const bool useSD_2 = pick_option(&argc, argv, "useSD_wien", NULL) != NULL;
  const bool verbose = pick_option(&argc, argv, "verbose", NULL) != NULL;

	//! Check parameters
	const unsigned tau_2D_hard  = (strcmp(_tau_2D_hard, "dct" ) == 0 ? DCT :
                                 (strcmp(_tau_2D_hard, "bior") == 0 ? BIOR : NONE));
    if (tau_2D_hard == NONE) {
        cout << "tau_2d_hard is not known." << endl;
        argc = 0; //abort
    }
	const unsigned tau_2D_wien  = (strcmp(_tau_2D_wien, "dct" ) == 0 ? DCT :
                                 (strcmp(_tau_2D_wien, "bior") == 0 ? BIOR : NONE));
    if (tau_2D_wien == NONE) {
        cout << "tau_2d_wien is not known." << endl;
        argc = 0; //abort
    };
	const unsigned color_space  = (strcmp(_color_space, "rgb"  ) == 0 ? RGB   :
                                 (strcmp(_color_space, "yuv"  ) == 0 ? YUV   :
                                 (strcmp(_color_space, "ycbcr") == 0 ? YCBCR :
                                 (strcmp(_color_space, "opp"  ) == 0 ? OPP   : NONE))));
    if (color_space == NONE) {
        cout << "color_space is not known." << endl;
        argc = 0; //abort
    };

  const int patch_size = atoi(_patch_size);
    if (patch_size < 0)
    {
      cout << "The patch_size parameter must not be negative." << endl;
      return EXIT_FAILURE;
    } else {
      const unsigned patch_size = (unsigned) patch_size;
    }
  const int nb_threads = 4;//atoi(_nb_threads);
    if (nb_threads < 0)
    {
      cout << "The nb_threads parameter must not be negative." << endl;
      return EXIT_FAILURE;
    } else {
      const unsigned nb_threads = (unsigned) nb_threads;
    }

  //! Check if there is the right call for the algorithm

  if (argc < 4) {
    cerr << "usage: " << argv[0] << " input sigma output [basic]\n\
             [-tau_2d_hard {dct,bior} (default: bior)]\n\
             [-useSD_hard]\n\
             [-tau_2d_wien {dct,bior} (default: dct)]\n\
             [-useSD_wien]\n\
             [-color_space {rgb,yuv,opp,ycbcr} (default: opp)]\n\
             [-patch_size {0,8,...} (default: 0, auto size, 8 or 12 depending on sigma)]\n\
             [-nb_threads (default: 0, auto number)]\n\
             [-verbose]" << endl;
    return EXIT_FAILURE;
  }

	//! Declarations
	vector<float> img_noisy, img_basic, img_denoised;
    unsigned width, height, chnls;

    //! Load image

	if(load_image("lena_noisy.png", img_noisy, &width, &height, &chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;

//saveChannelToCSV1("simage1.csv", img_noisy, width, height, 0, chnls);







//saveChannelToCSV1("simage2.csv", img, width, height, 0, chnls);




   // load_image("cinput.png", img_noisy, &width, &height, &chnls);



vector<float> noisy1=imageToVector("lena_noisy.png");


	float fSigma = atof(argv[2]);
   //! Denoising
   if (run_bm3d(35, img_noisy, img_basic, img_denoised, width, height, chnls,
                 useSD_1, useSD_2, tau_2D_hard, tau_2D_wien, YUV, patch_size,
                 nb_threads, verbose)
        != EXIT_SUCCESS)
        return EXIT_FAILURE;

   //! save noisy, denoised and differences images
   cout << endl << "Save images...";


 cout<<"1 "<<get_psnr(img_noisy, img_basic, width* height* chnls, 255)<<endl;
 cout<<"1 "<<get_psnr(img_noisy, img_denoised, width* height* chnls, 255)<<endl;


   if (argc > 4)
   if (save_image(argv[4], img_basic, width, height, chnls) != EXIT_SUCCESS)
      return EXIT_FAILURE;

	if (save_image(argv[3], img_denoised, width, height, chnls) != EXIT_SUCCESS)
		return EXIT_FAILURE;

    cout << "done." << endl;

	return EXIT_SUCCESS;
}

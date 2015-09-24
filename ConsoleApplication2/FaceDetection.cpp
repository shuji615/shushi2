#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#endif

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include <iostream>
#include <time.h>

#ifdef _DEBUG
#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
#pragma comment(lib, "opencv_imgproc249d.lib")
#pragma comment(lib, "opencv_video249d.lib")
#pragma comment(lib, "opencv_videostab249d.lib")
#pragma comment(lib, "opencv_superres249d.lib")
#pragma comment(lib, "opencv_ocl249d.lib")
#pragma comment(lib, "opencv_gpu249d.lib")
#pragma comment(lib, "opencv_objdetect249d.lib")
#else
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_highgui249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
#pragma comment(lib, "opencv_video249.lib")
#pragma comment(lib, "opencv_videostab249.lib")
#pragma comment(lib, "opencv_superres249.lib")
#pragma comment(lib, "opencv_ocl249.lib")
#pragma comment(lib, "opencv_gpu249.lib")
#pragma comment(lib, "opencv_objdetect249.lib")
#endif

#pragma comment(lib, "IlmImf.lib")
#pragma comment(lib, "libjasper.lib")
#pragma comment(lib, "libjpeg.lib")
#pragma comment(lib, "libpng.lib")
#pragma comment(lib, "libtiff.lib")
#pragma comment(lib, "zlib.lib")
#pragma comment(lib, "comctl32.Lib")
#pragma comment(lib,"vfw32.lib")  // "vfw32.lib"�ւ̃A�N�Z�X�𖾎��I�Ɏ����B�����"vfw.lib"�ɂ��A�N�Z�X�\�ɁB


#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include <opencv2/superres/superres.hpp>
#include <opencv2/core/core.hpp>        // core���W���[���̃w�b�_�[���C���N���[�h
#include <opencv2/highgui/highgui.hpp>  // highgui���W���[���̃w�b�_�[���C���N���[�h
#include <iostream>

#include "cv.h"
#include "highgui.h"

#include "stdafx.h"
#include "OpticalFlowIO.h"
using namespace std;

#include <time.h>

using namespace cv;
using namespace cv::superres;


int _main()
{
    cv::Mat img , gray;
	// ����t�@�C���̓ǂݍ���
	VideoCapture capture = VideoCapture("koizumi.mp4");

    //�J�������I�[�v���ł��Ȃ��ꍇ�I��
    if( !capture.isOpened() )
    {
        return -1;
    }
    
	/*
    // �E�B���h�E���쐬����
    char windowName[] = "camera";
    cv::namedWindow( windowName, CV_WINDOW_AUTOSIZE );
	*/
    
    // ���ފ�̓ǂݍ���(2��ނ��邩��D���ȕ���)
//    std::string cascadeName = "haarcascades/haarcascade_upperbody.xml";
    std::string cascadeName = "haarcascades/haarcascade_upperbody.xml";
    cv::CascadeClassifier cascade;
    if(!cascade.load(cascadeName))
        return -1;
    
    //scale�̒l��p���Č��摜���k���A�����Ȃ�8�r�b�g�����^�C1�`�����l��(���m�N��)�̉摜���i�[����z����쐬
    double scale = 4.0;    

	FILE *output;         // �o�̓X�g���[��
	output = fopen("face_detection.txt", "w");

    // �����L�[�����������܂ŁA���[�v������Ԃ�
    for(int count=0;;count++)
    {
        capture >> img;

		// �摜�f�[�^�擾�Ɏ��s�����烋�[�v�𔲂���
		if (img.empty()) break;

        // �O���[�X�P�[���摜�ɕϊ�
        cv::cvtColor(img, gray, CV_BGR2GRAY);
        cv::Mat smallImg(cv::saturate_cast<int>(img.rows/scale), cv::saturate_cast<int>(img.cols/scale), CV_8UC1);
        // �������ԒZ�k�̂��߂ɉ摜���k��
        cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
        cv::equalizeHist( smallImg, smallImg);
        
        std::vector<cv::Rect> faces;
        /// �}���`�X�P�[���i��j�T��xo
        // �摜�C�o�͋�`�C�k���X�P�[���C�Œ��`���C�i�t���O�j�C�ŏ���`
        cascade.detectMultiScale(smallImg, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
        //cascade.detectMultiScale(smallImg, faces);
        
		int facecount=0;

        // ���ʂ̕`��
        std::vector<cv::Rect>::const_iterator r = faces.begin();
        for(; r != faces.end(); ++r)
        {
			facecount++;
			
            cv::Point center;
            int radius;
            center.x = cv::saturate_cast<int>((r->x + r->width*0.5)*scale);
            center.y = cv::saturate_cast<int>((r->y + r->height*0.5)*scale);
            radius = cv::saturate_cast<int>((r->width + r->height)*0.25*scale);
            cv::circle( img, center, radius, cv::Scalar(80,80,255), 3, 8, 0 );

        }
		if(facecount>0){
			std::ostringstream oss;
			oss << "result_image/output_" << count << "_" << facecount << ".png";
			cv::imwrite( oss.str().c_str(),img );
		}
		fprintf(output,"%d\n",facecount);
		printf("%d\t%d\n",count,facecount);
        //cv::namedWindow("result", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
//        cv::imshow( windowName, img );
		
        //cv::waitKey(0);

    }
}
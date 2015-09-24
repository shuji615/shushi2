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
#pragma comment(lib,"vfw32.lib")  // "vfw32.lib"へのアクセスを明示的に示す。これで"vfw.lib"にもアクセス可能に。


#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include <opencv2/superres/superres.hpp>
#include <opencv2/core/core.hpp>        // coreモジュールのヘッダーをインクルード
#include <opencv2/highgui/highgui.hpp>  // highguiモジュールのヘッダーをインクルード
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
	// 動画ファイルの読み込み
	VideoCapture capture = VideoCapture("koizumi.mp4");

    //カメラがオープンできない場合終了
    if( !capture.isOpened() )
    {
        return -1;
    }
    
	/*
    // ウィンドウを作成する
    char windowName[] = "camera";
    cv::namedWindow( windowName, CV_WINDOW_AUTOSIZE );
	*/
    
    // 分類器の読み込み(2種類あるから好きな方を)
//    std::string cascadeName = "haarcascades/haarcascade_upperbody.xml";
    std::string cascadeName = "haarcascades/haarcascade_upperbody.xml";
    cv::CascadeClassifier cascade;
    if(!cascade.load(cascadeName))
        return -1;
    
    //scaleの値を用いて元画像を縮小、符号なし8ビット整数型，1チャンネル(モノクロ)の画像を格納する配列を作成
    double scale = 4.0;    

	FILE *output;         // 出力ストリーム
	output = fopen("face_detection.txt", "w");

    // 何かキーが押下されるまで、ループをくり返す
    for(int count=0;;count++)
    {
        capture >> img;

		// 画像データ取得に失敗したらループを抜ける
		if (img.empty()) break;

        // グレースケール画像に変換
        cv::cvtColor(img, gray, CV_BGR2GRAY);
        cv::Mat smallImg(cv::saturate_cast<int>(img.rows/scale), cv::saturate_cast<int>(img.cols/scale), CV_8UC1);
        // 処理時間短縮のために画像を縮小
        cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
        cv::equalizeHist( smallImg, smallImg);
        
        std::vector<cv::Rect> faces;
        /// マルチスケール（顔）探索xo
        // 画像，出力矩形，縮小スケール，最低矩形数，（フラグ），最小矩形
        cascade.detectMultiScale(smallImg, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
        //cascade.detectMultiScale(smallImg, faces);
        
		int facecount=0;

        // 結果の描画
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
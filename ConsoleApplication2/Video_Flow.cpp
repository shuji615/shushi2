﻿#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include <iostream>
#include <time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include <opencv2/superres/superres.hpp>
#include <opencv2/core/core.hpp>        // coreモジュールのヘッダーをインクルード
#include <opencv2/highgui/highgui.hpp>  // highguiモジュールのヘッダーをインクルード

#include <fstream>
#include <cstring>

#include <iterator>
#include <vector>
#include <algorithm>


#ifdef _DEBUG
#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
#pragma comment(lib, "opencv_imgproc249d.lib")
#pragma comment(lib, "opencv_video249d.lib")
#pragma comment(lib, "opencv_videostab249d.lib")
#pragma comment(lib, "opencv_superres249d.lib")
#pragma comment(lib, "opencv_ocl249d.lib")
#pragma comment(lib, "opencv_gpu249d.lib")
#else
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_highgui249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
#pragma comment(lib, "opencv_video249.lib")
#pragma comment(lib, "opencv_videostab249.lib")
#pragma comment(lib, "opencv_superres249.lib")
#pragma comment(lib, "opencv_ocl249.lib")
#pragma comment(lib, "opencv_gpu249.lib")
#endif

#pragma comment(lib, "IlmImf.lib")
#pragma comment(lib, "libjasper.lib")
#pragma comment(lib, "libjpeg.lib")
#pragma comment(lib, "libpng.lib")
#pragma comment(lib, "libtiff.lib")
#pragma comment(lib, "zlib.lib")
#pragma comment(lib, "comctl32.Lib")
#pragma comment(lib,"vfw32.lib")  // "vfw32.lib"へのアクセスを明示的に示す。これで"vfw.lib"にもアクセス可能に。

#define MAveWidth 100
#define FPS 10.0


#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cv.h"
#include "highgui.h"

#include "stdafx.h"
#include "OpticalFlowIO.h"
using namespace std;

#include <time.h>

#define Res 400
#define FRAME 30000

using namespace cv;
using namespace cv::superres;
using namespace std;


class Shot{
public:
	double StartTime;
	double EndTime;
	double FlowMinTime;
	double FlowMinValue;
	
	// 比較関数
    static bool cmp(Shot a, Shot b)
    {
		return (a.FlowMinValue < b.FlowMinValue); 
    } 
};


void OpenCVDefaultFlow(char *filename);
void CLFlow(char *filename);
void VisualizeDimageFlow(DImage & vx, DImage & vy, DImage & flow, char* filename);
int OpenCVSampleFlow(char* filename);
int OpticalFlowBasedCutandDefinePriority(char* filename);
void DetectSpeechArea(char* filename);
bool search_areas(Shot speecharea, double time);
Shot AdjustCutArea(Shot input, vector<Shot> speecharea);
vector<Shot> MakeSpeechAreaVector(char* filename);

int main(){
	int type;
	cout << "モード選択" << endl << "1 : OpenCVDefaultFlow" << endl << "2 : CLFlow" << endl << "3 : OpenCVSampleFlow" << endl << "4 : CutFromText" << endl << "5 : SpeechDetection" << endl;
	scanf("%d",&type);
	switch (type)
	{
	case 1:
		OpenCVDefaultFlow("test_data.mp4");
		break;
	case 2:
		CLFlow("yoshida_mini_all.mp4");
		break;
	case 3:
		OpenCVSampleFlow("test_data.mp4");
		break;
	case 4:
		OpticalFlowBasedCutandDefinePriority("yoshida");
		break;
	case 5:
		DetectSpeechArea("yoshida");
		break;
	case 6:
		MakeSpeechAreaVector("yoshida");
		break;
	default:
		break;
	}
}

void OpenCVDefaultFlow(char *filename){

	// 動画ファイルの読み込み
	VideoCapture capture = VideoCapture(filename);
	// TV-L1アルゴリズムによるオプティカルフロー計算オブジェクトの生成
	Ptr<DenseOpticalFlowExt> opticalFlow = superres::createOptFlow_DualTVL1();


	// 前のフレームを保存しておく
	Mat img1,img2;
	Mat prev,curr;
	capture >> img1;
	cv::cvtColor(img1, prev, CV_BGR2GRAY);


	int opt_hist[400];			//出力する，１フレームごとのオプティカルフローのヒストグラム

	FILE *output_hist;			//ヒストグラムを１行ずつ出力
	output_hist = fopen("flow_test_hist.txt", "w");

	FILE *output_sum;			//画像全体のオプティカルフローをフレームごとに出力
	output_sum = fopen("flow_test_sum.txt", "w");

	//////////デバッグ用
	//	FILE *output_debug1;         
	//	output_debug1 = fopen("flow_test_debug1.txt", "w");
	//	FILE *output_debug2;         
	//	output_debug2 = fopen("flow_test_debug2.txt", "w");
	//////////デバッグ用

	for (int count=0;;count++)
	{
		cout << "OpenCVFlow Frame: " << count  << "\n";

		for(int i=0;i<400;i++){
			opt_hist[i]=0;
		}
		// 現在のフレームを保存
		capture >> img2;
		cv::cvtColor(img2, curr, CV_BGR2GRAY);

		// 画像データ取得に失敗したらループを抜ける
		if (curr.empty()) break;

		// オプティカルフローの計算
		Mat flowX, flowY,flowXY;
		opticalFlow->calc(prev, curr, flowX, flowY);
//		opticalFlow->calc(prev, curr, flowXY);

		///////////↓デバッグ用↓
		/*
		Mat magnitude, angle;
		cartToPolar(flowX, flowY, magnitude, angle, true);

		Mat hsvPlanes[3];		
		hsvPlanes[0] = angle;
		normalize(magnitude, magnitude, 0, 1, NORM_MINMAX); // 正規化
		hsvPlanes[1] = magnitude;
		hsvPlanes[2] = Mat::ones(magnitude.size(), 5);
		//  HSVを合成して一枚の画像にする
		Mat hsv;
		merge(hsvPlanes, 3, hsv);
		//  HSVからBGRに変換
		Mat flowBgr;
		cvtColor(hsv, flowBgr, cv::COLOR_HSV2BGR);

		imshow("prev",prev);
		imshow("curr",curr);

		imshow("magnitude",magnitude);

		imshow("vx",flowX);
		imshow("vy",flowY);

		imshow("flowBgr",flowBgr);
		cvWaitKey(0);
		*/
		///////////↑デバッグ用↑

		double flow_sum=0;		//フレームごとのオプティカルフローの合計値

		for(int y = 0; y < flowX.rows; ++y){
			for(int x = 0; x < flowX.cols; ++x){
				double flow = sqrt(pow(flowX.at<unsigned char>(y, x),2) + pow(flowY.at<unsigned char>(y, x),2));
				//				double flow = sqrt(pow(flowX.data[y*flowX.cols+x],2) + pow(flowY.data[y*flowY.cols+x],2));
				//				double flow = magnitude.data[y*magnitude.cols+x];
				int k = std::min(399,(int)flow);
				opt_hist[k]++;
				flow_sum += flow;
				//				fprintf(output_debug1,"%lf\t",flowX.data[y*magnitude.cols+x]);
				//				fprintf(output_debug2,"%lf\t",flowY.data[y*magnitude.cols+x]);

			}
			//			fprintf(output_debug1,"\n");
			//			fprintf(output_debug2,"\n");
		}

		fprintf(output_sum ,"%lf\n", flow_sum );

		for(int j=0;j<400;j++){
			fprintf(output_hist, "%d\t",opt_hist[j] );
		}
		fprintf(output_hist, "\n");

		curr.copyTo(prev);

	}
}

void CLFlow(char* filename){

	// 動画ファイルの読み込み
	VideoCapture capture = VideoCapture(filename);
	// TV-L1アルゴリズムによるオプティカルフロー計算オブジェクトの生成
	Ptr<DenseOpticalFlowExt> opticalFlow = superres::createOptFlow_DualTVL1();

	// 前のフレームを保存しておく
	Mat prev,curr;
	capture >> prev;

	FILE *output_hist;
	output_hist = fopen("CLFlow_hist.txt", "w");

	FILE *output_sum;
	output_sum = fopen("CLFlow_sum.txt", "w");

	for (int count=0;;count++)
	{
		cout << "CLFlow Frame: " << count  << "\n";

		int opt_hist[1000];
		for(int i=0;i<1000;i++){
			opt_hist[i]=0;
		}

		// 現在のフレームを保存
		capture >> curr;

		// 画像データ取得に失敗したらループを抜ける
		if (curr.empty()) break;

		////
		OpticalFlowIO ofio;
		DImage Im1,Im2,vx,vy,warp;

		Im1.matimread(prev);
		Im2.matimread(curr);

		ofio.OpticalFlowIOMain(vx,vy,Im1,Im2,warp);

		/*
		DImage tmp;
		std::ostringstream oss_;
		oss_ << "arrow/arrow_" << count << ".jpg"; 
		VisualizeDimageFlow(vx, vy, tmp, const_cast<char*>( oss_.str().c_str()) );
		vx.imwrite("vx.jpg");
		vy.imwrite("vy.jpg");
		*/

		int opt_sum=0;
		for(int y = 0; y < vx.height(); ++y){
			for(int x = 0; x < vx.width(); ++x){
				int k = std::min(999,(int)(100*sqrt(pow(vx.pData[y*vx.width()+x],2) + pow(vy.pData[y*vx.width()+x],2))));
				opt_hist[k]++;
				opt_sum += k;
			}
		}
		fprintf(output_sum,"%d\n",opt_sum);

		for(int j=0;j<1000;j++){
			fprintf(output_hist, "%d\t",opt_hist[j]);
		}
		fprintf(output_hist, "\n");

		curr.copyTo(prev);
	}

}


// convert flow (vx, vy) into color image
void VisualizeDimageFlow(DImage & vx, DImage & vy, DImage & flow, char* filename)
{
	int width, height;
	width = vx.width();
	height = vx.height();

	flow.allocate(width, height, 3);	// color image

	double *uflow = new double[width*height];	// allocate
	double *vflow = new double[width*height];

	double unknown_flow_thresh = 1e9;
	double unknown_flow = 1e10;
	double maxu = -999;
	double maxv = -999;
	double minu = 999;
	double minv = 999;
	double maxrad = -1;
	double eps = 1e-7;

	// find max & min of each flow and radian
	for(int i=0; i<width*height; i++){
		if(vx[i] > maxu) maxu = vx[i];
		if(vx[i] < minu) minu = vx[i];
		if(vy[i] > maxv) maxv = vy[i];
		if(vy[i] < minv) minv = vy[i];

		double rad = sqrt(vx[i]*vx[i] + vy[i]*vy[i]);	// the strength of flow
		if(maxrad < rad) maxrad = rad;
	}

	// compute u v to relative values
	for(int i=0; i<width*height; i++){
		uflow[i] = vx[i]/(maxrad+eps);
		vflow[i] = vy[i]/(maxrad+eps);
	}

	// fix unknown flow
	for(int i=0; i<width*height; i++){
		if(fabs((double)(vx[i])) > unknown_flow_thresh || fabs((double)(vy[i])) > unknown_flow_thresh){
			uflow[i] = 0;
			vflow[i] = 0;
		}
	}

	// compute color
	ColorWheel cw;
	cw.makecolorwheel();
	for(int i=0; i<width*height; i++){

		unsigned char pix[3];
		float fx = uflow[i];
		float fy = vflow[i];
		cw.computeColor(fx, fy, pix);

		for(int k=0; k<3; k++){
			flow[i*3+k] = pix[k];
		}
	}


	// visualize arrow
	CvSize size;
	size.width = width;
	size.height = height;
	IplImage *image = cvCreateImage(size, IPL_DEPTH_8U, 3);
	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){
			for(int k=0; k<3; k++){
				((UINT8 *)(image->imageData + image->widthStep*y))[x*image->nChannels +k] = flow[(y*width+x)*3+k];
			}
		}
	}
	int arrow_interval=20;
	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){

			float fx = uflow[y*width+x];
			float fy = vflow[y*width+x];
			float rad = sqrt(fx*fx + fy*fy);

			if(rad > 0.2){

				if(x%arrow_interval==0 && y%arrow_interval==0){

					CvPoint2D32f point1, point2;
					point1.x = x;
					point1.y = y;

					point2.x = x+vx[y*width+x];
					point2.y = y+vy[y*width+x];

					//					cvLine(image, cvPointFrom32f(point1), cvPointFrom32f(point2), CV_RGB(255, 0, 0), 1, CV_AA, 0);
				}
			}
		}
	}
	cvSaveImage(filename, image);
	cvShowImage("arrow",image);
	cvWaitKey(0);
	cvReleaseImage(&image);


	delete [] uflow;	// release
	delete [] vflow;
}

#define OPENCV_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))
#define OPENCV_VERSION_CODE OPENCV_VERSION(CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION)

int OpenCVSampleFlow(char* filename)
{
	// 動画ファイルの読み込み
	VideoCapture capture = VideoCapture(filename);

	// image1
	cv::Mat img1,img2;
	cv::Mat prev, next;

	capture >> img1;
	cv::cvtColor(img1, prev, CV_BGR2GRAY);

	for(int count = 0;;count++){
		capture >> img2;
		cv::cvtColor(img2, next, CV_BGR2GRAY);

		std::vector<cv::Point2f> prev_pts;
		std::vector<cv::Point2f> next_pts;

		// 初期化
		cv::Size flowSize(30,30);
		cv::Point2f center = cv::Point(prev.cols/2., prev.rows/2.);
		for(int i=0; i<flowSize.width; ++i) {
			for(int j=0; j<flowSize.width; ++j) {
				cv::Point2f p(i*float(prev.cols)/(flowSize.width-1), 
					j*float(prev.rows)/(flowSize.height-1));
				prev_pts.push_back((p-center)*0.9f+center);
			}
		}

		// Lucas-Kanadeメソッド＋画像ピラミッドに基づくオプティカルフロー
		// parameters=default
#if OPENCV_VERSION_CODE > OPENCV_VERSION(2,3,0)
		cv::Mat status, error;
#else
		std::vector<uchar> status;
		std::vector<float> error;
#endif
		cv::calcOpticalFlowPyrLK(prev, next, prev_pts, next_pts, status, error);

		// オプティカルフローの表示
		std::vector<cv::Point2f>::const_iterator p = prev_pts.begin();
		std::vector<cv::Point2f>::const_iterator n = next_pts.begin();
		for(; n!=next_pts.end(); ++n,++p) {
			cv::line(img2, *p, *n, cv::Scalar(150,0,0),2);
		}

		cv::namedWindow("optical flow", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
		cv::imshow("optical flow", img2);
		cv::waitKey(0);
	}
	next.copyTo(prev);

}


int OpticalFlowBasedCutandDefinePriority(char* filename){

	ostringstream oss;
	oss << "digestmeta\\" << filename << "\\" << filename << "_CLFlow_sum.txt";
	std::ifstream ifs(oss.str().c_str());
    if (ifs.fail())
    {
        std::cerr << "失敗" << std::endl;
        return -1;
    }
 
	std::vector<double> RawData;
	std::vector<double> MAveData;
	std::copy(std::istream_iterator<double>(ifs), std::istream_iterator<double>(), std::back_inserter(RawData));

	//MAveWidth 幅で平均化し，MAveDataに格納
	//同時にMAveDataの平均の計算
	double average=0;
	for(int i=0;i<RawData.size()-MAveWidth;i++){
		for(int j=0;j<MAveWidth;j++){
			if(j==0){
				MAveData.push_back(RawData[i+j]);
			}else{
				MAveData[i] += RawData[i+j];
			}
		}
		MAveData[i] /= MAveWidth;
		average += MAveData[i];
	}
	average /= MAveData.size();
	const double THRE = 0.7 * average;

	std::vector<Shot> Shots;
	Shot tmp;

	bool flag = false;
	for(int i=0;i<MAveData.size();i++){
		if(flag == true){
			if(MAveData[i] < tmp.FlowMinValue){
				tmp.FlowMinValue = MAveData[i];
				tmp.FlowMinTime = (double)i/FPS;
			}
			if(MAveData[i] > THRE){
				tmp.EndTime = (double)i/FPS;
				Shots.push_back(tmp);

				flag = false;
			}
		}else if(MAveData[i] < THRE){
			tmp.StartTime = (double)i/FPS;
			tmp.FlowMinTime = (double)i/FPS;
			tmp.FlowMinValue = MAveData[i];

			flag = true;
		}
	}
	std::sort(Shots.begin(), Shots.end(), &Shot::cmp);

	vector<Shot> cut_areas;

	for(int i=0;i<10;i++){
		Shot tmp;
		tmp.StartTime = Shots[i].FlowMinTime;
		tmp.EndTime = Shots[i].FlowMinTime + 10.0 ;
		cut_areas.push_back(tmp);

		std::cout << (int)cut_areas[i].StartTime/60  << " : " << (int)cut_areas[i].StartTime%60 << " - " << (int)cut_areas[i].EndTime/60 << " : " << (int)cut_areas[i].EndTime%60 << endl;
	}

	DetectSpeechArea(filename);
	for each(Shot shot in cut_areas){
		vector<Shot> speecharea = MakeSpeechAreaVector(filename);
		shot = AdjustCutArea(shot,speecharea);
	}

	for(int i=0;i<10;i++){
		std::cout << (int)cut_areas[i].StartTime/60  << " : " << (int)cut_areas[i].StartTime%60 << " - " << (int)cut_areas[i].EndTime/60 << " : " << (int)cut_areas[i].EndTime%60 << endl;
	}
	scanf("%s");
    return 0;
}

void DetectSpeechArea(char* filename){

	int ret;
	std::ostringstream makemonofile;
	makemonofile << "ffmpeg -i digestmeta\\" << filename << "\\" << filename << ".wav";
	makemonofile << " -ac 1 digestmeta\\" << filename << "\\" << filename << "_mono.wav";
	ret = system(makemonofile.str().c_str());
	if (ret != 0){
		printf("モノラルファイル作成失敗\n");
	}

	FILE *filelist;
	std::ostringstream filelist_name;
	filelist_name << "digestmeta\\" << filename << "\\" << filename << "_filelist.txt";

	std::ostringstream filelist_contents;
	filelist_contents << "digestmeta\\" << filename << "\\" << filename << "_mono.wav";

	filelist = fopen(filelist_name.str().c_str(), "w");
	fprintf(filelist,filelist_contents.str().c_str());
	fclose(filelist);

	std::ostringstream run_adintool;
	run_adintool << "adintool.exe -in file -filelist " << filelist_name.str().c_str() << " -out file -filename adintool_result\\ -freq 48000 -headmargin 1000 -tailmargin 1000 -lv 15000 > ";
	run_adintool << "digestmeta\\" << filename << "\\" << filename << "_speechareas.txt";
	ret = system(run_adintool.str().c_str());
	if (ret != 0){
		printf("Adintool起動失敗\n");
	}
}


bool search_areas(Shot speecharea, double time){
	if(speecharea.StartTime<time && time<speecharea.EndTime){
		return true;
	}else{
		return false;
	}
}

Shot AdjustCutArea(Shot input, vector<Shot> speecharea){
	Shot out;
	double starttime = input.StartTime;
	double endtime = input.EndTime;
	out.StartTime = starttime;
	out.EndTime = endtime;

	for(int i=0;i<speecharea.size();i++){
		if(search_areas(speecharea[i],starttime) == true){
			out.StartTime = speecharea[i].StartTime;
			break;
		}
	}
	for(int i=0;i<speecharea.size();i++){
		if(search_areas(speecharea[i],endtime) == true){
			out.EndTime = speecharea[i].EndTime;
			break;
		}
	}
	return out;
}

vector<Shot> MakeSpeechAreaVector(char* filename){
	vector<Shot> out;
	Shot tmp;
	
	ostringstream oss;
	oss << "digestmeta\\" << filename << "\\" << filename << "_speechareas.txt";

    ifstream file(oss.str().c_str());
	string temp;
    vector<string> items;

	while(getline(file, temp, '['))
    {
        items.push_back(temp);
    }
	for(int i=0;i<items.size();i++){
		if(i>0){
			char s2[] = "(s)";
			char *tok;
			char *end;

			tok = strtok( (char*)items[i].c_str(), s2 );
			for(int j=0;j<4 && tok!=NULL;j++){
				if(j==1){
					tmp.StartTime = strtod(tok,&end);
				}
				if(j==3){
					tmp.EndTime = strtod(tok,&end);
				}
				tok = strtok( NULL, s2 );
			}
			out.push_back(tmp);
		}
	}
	return out;	
}
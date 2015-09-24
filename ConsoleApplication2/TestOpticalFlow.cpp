// TestOpticalFlow.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "OpticalFlowIO.h"
using namespace std;

#include <time.h>


//
// 2013 7/8 written by Rei Kawakami based on Ce Liu's code. All rights resereved.
// This code is made for Goto-san's research
// Please cite the paper if you are using this code in your research
// http://people.csail.mit.edu/celiu/ECCV2008/
//

// #define debug 1
//↑のコメントアウトを外すとOK

/*
int _tmain(int argc, _TCHAR* argv[])
{
	 clock_t start,end;
	 start = clock();
	
	DImage Im1,Im2,warp;
	char *filename[3];
#ifdef debug
	filename[0]="IMG_a.jpg";//"Epson_0737_3.png";//"test064.jpg"; //"test009.jpg"; //"car1.jpg";
	filename[1]="IMG_b.jpg";//"Epson_0853_5.png";//"test065.jpg"; //"test010.jpg"; //"car2.jpg";
	filename[2]="warp.png";
#else
	if(argc != 4){
		cout << "TestOpticalFlow.exe [image1] [image2] [warpedimage]\n";
		return 0;
	}else{
		filename[0] = argv[1];
		filename[1] = argv[2];
		filename[2] = argv[3];
	}
#endif
	Im1.imread(filename[0]);
	Im2.imread(filename[1]);
	if(Im1.matchDimension(Im2)==false)
		cout << "The two images don't match!\n";
	
	DImage vx,vy;
	OpticalFlowIO ofio;
	ofio.OpticalFlowIOMain(vx, vy, Im1, Im2, warp);

	warp.imwrite(filename[2]);

	end = clock();
	printf("processing time: %.2f second.\n",(double)(end-start)/CLOCKS_PER_SEC);
	
	// output the parameters
	//vx.imwrite("output1.jpg");
	//vy.imwrite("output2.jpg");
	
	return 0;
}

*/
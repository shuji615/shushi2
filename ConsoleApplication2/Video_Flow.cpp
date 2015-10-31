#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include <iostream>
#include <time.h>
#include <direct.h>
#include <sys/stat.h>
#include <fstream>
#include <cstring>
#include <iterator>
#include <vector>
#include <algorithm>


#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include <opencv2/superres/superres.hpp>
#include <opencv2/core/core.hpp>        // coreモジュールのヘッダーをインクルード
#include <opencv2/highgui/highgui.hpp>  // highguiモジュールのヘッダーをインクルード
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>


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

#define MAveWidth 110 //10(fps) * 15(秒)
#define FPS 10.0
#define Res 400
#define FRAME 30000
#define OpticalFlowUnCuttableRate 0.3
#define VolumeSmoothRange 1
#define VolumeAveRange 0.01//秒単位
#define FLOWTHRE 1.3
#define AdinVolumeThreshold 4500
#define VolumeThre 250
#define BestN 10
#define MaxTime 15

#include "cv.h"
#include "highgui.h"
#include "stdafx.h"
#include "OpticalFlowIO.h"
#include "wave.h"

using namespace std;
using namespace cv;
using namespace cv::superres;

class Shot{
public:
	double StartTime;
	double EndTime;
	double FlowMinTime;
	double FlowMinValue;
	double VolumeAverage;
	double FlowAverage;
	double FlowDirection;
	double VolumeChangeSum;
	double VolumeAndFlowMixPriority;
	double VolumeChangeThreUnderSum;
	double VolumeChangeThreOverSum;
	bool ExclusionFlag;

	
	// 比較関数
    static bool flow_cmp(Shot a, Shot b)
    {
		return (a.FlowMinValue < b.FlowMinValue); 
    } 
    static bool flowAve_cmp(Shot a, Shot b)
    {
		return (a.FlowAverage < b.FlowAverage); 
    }

	static bool volumeAve_cmp(Shot a, Shot b)
    {
		return (a.VolumeAverage > b.VolumeAverage); 
    } 

	static bool VolumeAndFlowMixPriority_cmp(Shot a, Shot b)
    {
		return (a.VolumeAndFlowMixPriority > b.VolumeAndFlowMixPriority); 
    }
	static bool StartTime_cmp(Shot a, Shot b)
    {
		return (a.StartTime < b.StartTime); 
    }
	static bool VolumeChangeThreOverSum_cmp(Shot a, Shot b)
    {
		return (a.VolumeChangeThreOverSum > b.VolumeChangeThreOverSum); 
    }
	static bool VolumeChangeThreUnderSum_cmp(Shot a, Shot b)
    {
		return (a.VolumeChangeThreUnderSum > b.VolumeChangeThreUnderSum); 
    }

	void set_FlowAverage(double in){FlowAverage = in;}
	void set_VolumeAverage(double in){VolumeAverage = in;}
	void set_VolumeChangeSum(double in){VolumeChangeSum = in;}
	void set_VolumeChangeThreOverSum(double in){VolumeChangeThreOverSum = in;}
	void set_VolumeChangeThreUnderSum(double in){VolumeChangeThreUnderSum = in;}
	void set_StartTime(double in){StartTime = in;}
	void set_EndTime(double in){EndTime = in;}
	void set_FlowDirection(double in){FlowDirection = in;}
	void set_VolumeAndFlowMixPriority(double in){VolumeAndFlowMixPriority = in;}

};

class sukima{
public:
	double Starttime;
	double Endtime;
	double Widthtime;

	static bool Widthtime_cmp(sukima a, sukima b)
    {
		return (a.Widthtime > b.Widthtime); 
    }

};



void OpenCVDefaultFlow(char *filename);
void CLFlow(char *filename);
void VisualizeDimageFlow(DImage & vx, DImage & vy, DImage & flow, char* filename);
int OpenCVSampleFlow(char* filename);
vector<Shot> MakeShotBaseByOpticalFlow(char* filename);
vector<Shot> MakeShotBaseBySpeech(char* filename,double speechinterval, int shotmaxtime);
int OpticalFlowBasedCutandAdjustShots(char* filename);
void DetectSpeechArea(char* filename, int volumethreshold, int speechinterval);
bool search_areas(Shot speecharea, double time);
vector<Shot> SpeechBasedCut(char* filename, int volumethreshold, int speechinterval);
vector<Shot> MakeOpticalFlowBasedUncuttableAreaVector(char* filename);
int SoundBasedAreaCut(char* filename);
vector<short> readwave(char* filename);
int samplingrateofwave(char* filename);
vector<double> CalcMoveAve(vector<double> inputvector, int Range);
vector<double> CalcMoveAve(vector<short> inputvector, int Range);
vector<double> CalcAve(vector<short> inputvector, int Range);
double ClacOpticalFlowMoveDirection(DImage vx, DImage vy);
void CalcFlowAve(vector<Shot> &speecharea, char* filename);
void CalcVolumeChangeAveandSum(vector<Shot> &shots, char* filename);
void CalcShotPriority(vector<Shot> &shots,char* filename, double volume_ratio, double optflow_ratio);
void CalcFlowDirection(vector<Shot> &shots, char* filename);
double CalcVolumeAveofWav(char* filename);
void JointSpeechAreas(vector<Shot> &speechareas, double speechinterval,double maxtime);
void CalcVolumeAndFlowMixPriority(vector<Shot> &shots, double volume_ratio, double optflow_ratio);
void CutAndDefinePriority(char* filename, int Base, double volume_ratio, double optflow_ratio, int adjustmode);
void MakeMonoFile(char* filename);
void tmp();
vector<Shot> CutSpeechArea(vector<Shot> &speechareas,double maxtime,int joint_start,int joint_end);
void AdjustCutAreaBySpeechArea(vector<Shot> &adjustingarea, vector<Shot> &continuityareas, int mode);
void CalcVolumeChangeAveandSum(vector<Shot> &shots, char* filename, double threshold);
void CutAndDefinePriority(char* filename);
void ExtendSpeechAreas(vector<Shot> &speechareas);
void MakeBestDigests(char* filename);
void CalcVolumeAndFlowMixPriority(vector<Shot> &shots, double volume_ratio,double volume_thre_ratio , double optflow_ratio);
void AdjustCutAreaBySpeechArea(vector<Shot> &adjustingarea, vector<Shot> &speecharea);
int SearchBoundarySpeech(double time , vector<Shot> &speecharea);


int main(int argc, char*argv[]){

	string filename;
	if(argc <2){
		cout << "ファイル名を入力してください" << endl;
		char tmp_name[50];
		scanf("%s",tmp_name);
		filename = tmp_name;
	}else{
		filename = argv[1];
	}
//	CLFlow((char*)filename.c_str());
	MakeBestDigests((char*)filename.c_str());
//	CutAndDefinePriority((char*)filename.c_str());
}

vector<double> movetime;

vector<Shot> MakeShotBaseByOpticalFlow(char* filename){
	//CLFlowの結果を基にオプティカルフローが小さい範囲をショットとして切り出す

	//CLFlowの結果がなければ作る
	ostringstream oss;
	oss << "digestmeta\\" << filename << "\\" << filename << "_CLFlow_sum.txt";
	std::ifstream ifs(oss.str().c_str());
    if (ifs.fail())
    {
		CLFlow(filename);
    }

	//CLFlowの結果をRawDataにいったん格納し，その後移動平均を取ってMAveDataに格納
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
	const double THRE = 1.0 * average;//MAveDataの平均を閾値と定義

	vector<Shot> StopArea;
	Shot tmp;
	//オプティカルフローがTHREより小さい区間を，１つの展示にとどまっている区間と定義
	//１区間の中で，最もオプティカルフローが小さい，(MAveWidth/fps)秒の区間を，オプティカルフローが小さい区間として出力
	//現在は，MAveWidth/fps = 11としている（これは，何秒単位でオプティカルフローが小さい区間を切り出すかによる）
	bool flag = false;
	for(int i=0;i<MAveData.size();i++){
		if(flag == true){
			if(MAveData[i] < tmp.FlowMinValue){
				tmp.FlowMinValue = MAveData[i];
				tmp.FlowMinTime = (double)i/FPS;
			}
			if(MAveData[i] > THRE){
				tmp.EndTime = (double)i/FPS;
				StopArea.push_back(tmp);
				movetime.push_back((double)i/FPS);

				flag = false;
			}
		}else if(MAveData[i] < THRE){
			tmp.StartTime = (double)i/FPS;
			tmp.FlowMinTime = (double)i/FPS;
			tmp.FlowMinValue = MAveData[i];

			flag = true;
		}
	}

	//出力
	vector<Shot> OpticalFlowMinArea;
	for(int i=0;i<StopArea.size();i++){
		Shot tmp;
		tmp.StartTime = StopArea[i].FlowMinTime;
		tmp.EndTime = StopArea[i].FlowMinTime + (MAveWidth/10) ;
		tmp.FlowMinValue = StopArea[i].FlowMinValue;
		OpticalFlowMinArea.push_back(tmp);
		
//		std::cout << (int)OpticalFlowMinArea[i].StartTime/60  << " : " << (int)OpticalFlowMinArea[i].StartTime%60 <<
//			" - " << (int)OpticalFlowMinArea[i].EndTime/60 << " : " << (int)OpticalFlowMinArea[i].EndTime%60 << endl;
	}


	if(OpticalFlowMinArea.size() > 40){
		std::sort(OpticalFlowMinArea.begin(), OpticalFlowMinArea.end(), &Shot::flow_cmp);
		OpticalFlowMinArea.erase(OpticalFlowMinArea.begin()+40,OpticalFlowMinArea.end());
		std::sort(OpticalFlowMinArea.begin(), OpticalFlowMinArea.end(), &Shot::StartTime_cmp);
	}

	return OpticalFlowMinArea;
}

vector<Shot> MakeShotBaseBySpeech(char* filename,double speechinterval, int shotmaxtime){

	MakeMonoFile(filename);
	double volumeave = CalcVolumeAveofWav(filename);
	//DetectSpeechArea(filename,8000+2*volumeave,400);
	DetectSpeechArea(filename,AdinVolumeThreshold,400);

	//発話区間をadintoolで計算し， speecharea に入れる
	//vector<Shot> speecharea = SpeechBasedCut(filename,8000+2*volumeave,400);
	vector<Shot> speecharea = SpeechBasedCut(filename,AdinVolumeThreshold,400);
	//発話間隔がspeechinterval秒以内だったら，一つにつなげる
	JointSpeechAreas(speecharea,speechinterval,shotmaxtime);
	//前の発話箇所まで，あるいは２秒間区間を広げる
	ExtendSpeechAreas(speecharea);

	return speecharea;
}
void MakeMonoFile(char* filename){

	int ret;
	std::ostringstream makemonofile,monofilename;
	monofilename << "digestmeta\\" << filename << "\\" << filename << "_mono.wav";

	FILE *fileexisttest;
	if ( (fileexisttest = fopen(monofilename.str().c_str(),"r")) != NULL ){
		fclose( fileexisttest );
		// ファイルが存在する
	}
	else{
		makemonofile << "ffmpeg -i digestmeta\\" << filename << "\\" << filename << ".mp4";
		makemonofile << " -ac 1 -af \"bandreject=f=500:width_type=h:w=200\" digestmeta\\" << filename << "\\" << filename << "_mono.wav";
		ret = system(makemonofile.str().c_str());
		if (ret != 0){
			printf("モノラルファイル作成失敗\n");
		}
	}

}

void MakeBestDigests(char* filename){
	//shot[1]：speech
	//shot[2]：OpticalFlow

	//ベース作り
	cout << "ベース作り開始" << endl;

	vector<Shot> shots[3],shots_mix;
	shots[0] = MakeShotBaseBySpeech(filename,2,30); //真ん中を11秒にして，最大でそれを２秒ずつ両端に伸ばしたMAX15秒の範囲をとってくる
	shots[1] = MakeShotBaseByOpticalFlow(filename);

	//範囲の調整
	cout << "範囲調整開始" << endl;

	//発話
	//7秒に満たないものは除外する
	for(int i=0;i<shots[0].size();){
		if( (shots[0][i].EndTime - shots[0][i].StartTime) < 7){
			shots[0].erase(shots[0].begin()+i);
		}else{
			i++;
		}
	}
	//CLFlow_sumから，FlowAverageを計算
	CalcFlowAve(shots[0],filename);

	//CLFlowの全体の平均と分散を計算
	ostringstream oss;
	oss << "digestmeta\\" << filename << "\\" << filename << "_CLFlow_sum.txt";
	std::ifstream ifs(oss.str().c_str());

	//CLFlowの結果をRawDataにいったん格納し，その後移動平均を取ってMAveDataに格納
	std::vector<double> RawData;
	std::vector<double> MAveData;
	std::copy(std::istream_iterator<double>(ifs), std::istream_iterator<double>(), std::back_inserter(RawData));

	double average=0, dispersion=0;
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

	for(int i=0;i<MAveData.size();i++){
		dispersion += pow( MAveData[i] - average , 2) ;
	}
	dispersion = sqrt(dispersion / MAveData.size() );

	const double THRE = average + dispersion;//MAveDataの平均から１σを閾値と定義		

	//FlowAverageを基に，あまりに動きが大きいところは区間から除外するようにする
	for(auto shot = shots[0].begin() ; shot !=shots[0].end() ; ){
		if(shot->FlowAverage > THRE){
			shots[0].erase(shot);
		}else{
			shot++;
		}
	}

	//オプティカルフローの移り変わりを元に，一つの立ち止まっている区間での発話ピックアップを１シーンまでに
	for(int i=0;i<movetime.size()-1;i++){
		int best;
		vector<int> nearscenes;
		for(int j=0;j<shots[0].size();j++){
			if( movetime[i] < shots[0][j].StartTime && shots[0][j].StartTime < movetime[i+1]){
				nearscenes.push_back(j);
			}
		}
		if(nearscenes.size()>=1){
			best = nearscenes[0];
			for(int k=1;k<nearscenes.size();k++){
				if(shots[0][best].VolumeChangeSum < shots[0][nearscenes[k]].VolumeChangeSum){
					best = nearscenes[k];
				}
			}
			shots[2].push_back(shots[0][best]);
		}
		nearscenes.clear();
	}

	/*
	//被っている時間が半分以上ある場合は，VolumeChangeAveandSumの値が大きい方を残して，小さい方を消す
	CalcVolumeChangeAveandSum(shots[0],filename);	//ショットのVolumeChangeSumを計算 .VolumeChangeSum
	for(int i=0;i<shots[0].size()-1;i++){
		if(shots[0][i].EndTime-2 > shots[0][i+1].StartTime){
			if(shots[0][i].VolumeChangeSum > shots[0][i+1].VolumeChangeSum){
				shots[0].erase(shots[0].begin()+i+1);
			}else{
				shots[0].erase(shots[0].begin()+i);
			}
		}
	}*/

	//オプティカルフロー

	//基礎的なspeechareaを作る
	MakeMonoFile(filename);
	double volumeave = CalcVolumeAveofWav(filename);
	//DetectSpeechArea(filename,8000+2*volumeave,400);
	DetectSpeechArea(filename,AdinVolumeThreshold,400);

	//発話区間をadintoolで計算し， speecharea に入れる
	//vector<Shot> speecharea = SpeechBasedCut(filename,8000+2*volumeave,400);
	vector<Shot> speecharea = SpeechBasedCut(filename,AdinVolumeThreshold,400);

	//speechareaから，発話の途中でショットがカットされないように調整
//	AdjustCutAreaBySpeechArea(shots[1],speecharea,1);
	AdjustCutAreaBySpeechArea(shots[1],speecharea);


	//7秒に満たないものは除外する
	for(int i=0;i<shots[1].size();){
		if( (shots[1][i].EndTime - shots[1][i].StartTime) < 7){
			shots[1].erase(shots[1].begin()+i);
		}else{
			i++;
		}
	}

	//被っている時間が2秒以上ある場合は，FlowAveの値が小さい方を残して，大きい方を消す
	CalcFlowAve(shots[1],filename);
	for(int i=0;i<shots[1].size()-1;){
		if(shots[1][i].EndTime-2 > shots[1][i+1].StartTime){
			if(shots[1][i].FlowAverage < shots[1][i+1].FlowAverage){
				shots[1].erase(shots[1].begin()+i+1);
			}else{
				shots[1].erase(shots[1].begin()+i);
			}
		}else{
			i++;
		}
	}
	
	//2つのカットを1つにする
	for(int i=0;i<shots[1].size();i++){
		shots_mix.push_back(shots[1][i]);
	}
	for(int i=0;i<shots[2].size();i++){
		shots_mix.push_back(shots[2][i]);
	}

	std::sort(shots_mix.begin(),shots_mix.end(),Shot::StartTime_cmp);

	//前後間隔が30秒以内の場合は，MixPriorityの値が大きい方を残す
	/*
	CalcVolumeChangeAveandSum(shots_mix,filename);	//ショットのVolumeChangeSumを計算 .VolumeChangeSum
	CalcFlowAve(shots_mix,filename);
	CalcVolumeAndFlowMixPriority(shots_mix,1,1,1);										//volume_ratio,optflow_ratioの比で優先度を定義
	for(int i=0;i<shots_mix.size()-1;){
		if(shots_mix[i].StartTime + 30 > shots_mix[i+1].StartTime){
			if(shots_mix[i].VolumeAndFlowMixPriority > shots_mix[i+1].VolumeAndFlowMixPriority){
				shots_mix.erase(shots_mix.begin()+i+1);
			}else{
				shots_mix.erase(shots_mix.begin()+i);
			}
		}else{
			i++;
		}
	}	
	*/

	//ソート
	cout << "ソート開始" << endl;
	CalcFlowAve(shots_mix,filename);				//ショットのオプティカルフローの平均を計算 .FlowAve
	CalcVolumeChangeAveandSum(shots_mix,filename);	//ショットのVolumeChangeSumを計算 .VolumeChangeSum
	//CalcFlowDirection(shots_mix,filename);			//ショットのFlowDirectionの平均を計算（使わなさそう）

	//パラメータ推定用
	/*
	FILE *output_d;
	ostringstream oss_d;
	oss_d << "digestmeta\\" << filename << "\\" << filename << "_d.txt";
	output_d = fopen(oss_d.str().c_str(), "w");

	for(int i=0;i<shots_mix.size();i++){
		ostringstream tmp;
		tmp << shots_mix[i].VolumeChangeSum << "\t" << shots_mix[i].VolumeChangeThreSum << "\t" << shots_mix[i].FlowAverage << endl ;
		fprintf(output_d,tmp.str().c_str());
	}
	fclose(output_d);
	*/
	//パラメータ推定用

	//自分用
//	CalcVolumeAndFlowMixPriority(shots_mix,1,1,0);										//volume_ratio,optflow_ratioの比で優先度を定義
//	std::sort(shots_mix.begin(), shots_mix.end(), &Shot::VolumeAndFlowMixPriority_cmp);		//実際にソートする関数
	CalcVolumeAndFlowMixPriority(shots[2],1,0,0);										//volume_ratio,optflow_ratioの比で優先度を定義
	std::sort(shots[2].begin(), shots[2].end(), &Shot::VolumeAndFlowMixPriority_cmp);		//実際にソートする関数

	/*
	//batに出力↓
	FILE *output_selfspeak;
	ostringstream oss_selfspeak;
	oss_selfspeak << "bat3\\" << filename << "_selfspeakdigest.bat";
	output_selfspeak = fopen(oss_selfspeak.str().c_str(), "w");
	*/

	vector<Shot> shots_selfspeak;
	double Digesttime_selfspeak=0;
//	for(int i=0;Digesttime_selfspeak<180;i++){//時間で個数を選ぶ場合
	for(int i=0;i<20;i++){//個数で選ぶ
		shots_selfspeak.push_back(shots[2][i]);
		Digesttime_selfspeak += (shots[2][i].EndTime - shots[2][i].StartTime);
	}
	//std::sort(shots_selfspeak.begin(), shots_selfspeak.end(), &Shot::StartTime_cmp);

	/*
	for(int i=0;i<shots_selfspeak.size();i++){
		ostringstream tmp;
		tmp << "ffmpeg -r 30 -i ../digestmeta\\大きい動画\\" << filename << "_QHD.mp4 -ss " << shots_selfspeak[i].StartTime
			<< " -t " << shots_selfspeak[i].EndTime - shots_selfspeak[i].StartTime << " -af \"afade=t=in:ss=0:d=1,afade=t=out:st=" << shots_selfspeak[i].EndTime-1
			<< ":d=1\" -vf \"fade=in:0:30,fade=out:" << (shots_selfspeak[i].EndTime-1)*30 <<":30\" ../digestmeta\\" << "途中ファイル" << "\\" << filename << "\\selfspeak_" << i << ".mp4" << endl ;
		fprintf(output_selfspeak,tmp.str().c_str());
	}
	fclose(output_selfspeak);
	*/
	//batに出力↑

	//Viewerのメタファイル↓
	FILE *output_selfspeak_txt;
	ostringstream oss_selfspeak_txt;
	oss_selfspeak_txt << "digestmeta\\" << filename << "\\" << filename << "_selfspeak_digest.txt";
	output_selfspeak_txt = fopen(oss_selfspeak_txt.str().c_str(), "w");

	for(int i=0;i<shots_selfspeak.size();i++){
		ostringstream tmp;
		tmp << shots_selfspeak[i].StartTime << "\t" << shots_selfspeak[i].EndTime << "\t\n";
		fprintf(output_selfspeak_txt,tmp.str().c_str());
	}
	fclose(output_selfspeak_txt);	
	//Viewerのメタファイル↑


	//他者の発話
	CalcVolumeAndFlowMixPriority(shots[2],0,1,0);										//volume_ratio,optflow_ratioの比で優先度を定義
	std::sort(shots[2].begin(), shots[2].end(), &Shot::VolumeAndFlowMixPriority_cmp);		//実際にソートする関数

	vector<Shot> shots_someonespeak;
	double Digesttime_someonespeak=0;
//	for(int i=0;Digesttime_someonespeak<180;i++){//時間で個数を選ぶ場合
	for(int i=0;i<20;i++){//個数で選ぶ
		shots_someonespeak.push_back(shots[2][i]);
		Digesttime_someonespeak += (shots[2][i].EndTime - shots[2][i].StartTime);
	}

	FILE *output_someonespeak_txt;
	ostringstream oss_someonespeak_txt;
	oss_someonespeak_txt << "digestmeta\\" << filename << "\\" << filename << "_someonespeak_digest.txt";
	output_someonespeak_txt = fopen(oss_someonespeak_txt.str().c_str(), "w");

	for(int i=0;i<shots_someonespeak.size();i++){
		ostringstream tmp;
		tmp << shots_someonespeak[i].StartTime << "\t" << shots_someonespeak[i].EndTime << "\t\n";
		fprintf(output_someonespeak_txt,tmp.str().c_str());
	}
	fclose(output_someonespeak_txt);	


	//オプティカルフロー
//	CalcVolumeAndFlowMixPriority(shots_mix,0,0,1);										//volume_ratio,optflow_ratioの比で優先度を定義
//	std::sort(shots_mix.begin(), shots_mix.end(), &Shot::VolumeAndFlowMixPriority_cmp);		//実際にソートする関数
	CalcVolumeAndFlowMixPriority(shots[1],0,0,1);										//volume_ratio,optflow_ratioの比で優先度を定義
	std::sort(shots[1].begin(), shots[1].end(), &Shot::VolumeAndFlowMixPriority_cmp);		//実際にソートする関数

	/*
	//batに出力↓
	FILE *output_optflow;
	ostringstream oss_optflow;
	oss_optflow << "bat3\\" << filename << "_optflowdigest.bat";
	output_optflow = fopen(oss_optflow.str().c_str(), "w");
	*/

	vector<Shot> shots_optflow;
	double Digesttime_optflow=0;
//	for(int i=0;Digesttime_optflow<60;i++){//時間で個数を選ぶ場合
	for(int i=0;i<20;i++){//個数で選ぶ
		shots_optflow.push_back(shots[1][i]);
		Digesttime_optflow += (shots[1][i].EndTime - shots[1][i].StartTime);
	}
	//std::sort(shots_optflow.begin(), shots_optflow.end(), &Shot::StartTime_cmp);

	/*
	for(int i=0;i<shots_optflow.size();i++){
		ostringstream tmp;
		tmp << "ffmpeg -r 30 -i ../digestmeta\\大きい動画\\" << filename << "_QHD.mp4 -ss " << shots_optflow[i].StartTime
			<< " -t " << shots_optflow[i].EndTime - shots_optflow[i].StartTime << " -af \"afade=t=in:ss=0:d=1,afade=t=out:st=" << shots_optflow[i].EndTime-1
			<< ":d=1\" -vf \"fade=in:0:30,fade=out:" << (shots_optflow[i].EndTime-1)*30 <<":30\" ../digestmeta\\" << "途中ファイル" << "\\" << filename << "\\optflow_" << i << ".mp4" << endl ;
		fprintf(output_optflow,tmp.str().c_str());
	}

	fclose(output_optflow);
	*/
	//batに出力↑

	//Viewerのメタファイル↓
	FILE *output_optflow_txt;
	ostringstream oss_optflow_txt;
	oss_optflow_txt << "digestmeta\\" << filename << "\\" << filename << "_optflow_digest.txt";
	output_optflow_txt = fopen(oss_optflow_txt.str().c_str(), "w");

	for(int i=0;i<shots_optflow.size();i++){
		ostringstream tmp;
		tmp << shots_optflow[i].StartTime << "\t" << shots_optflow[i].EndTime << "\t\n";
		fprintf(output_optflow_txt,tmp.str().c_str());
	}
	fclose(output_optflow_txt);	
	//Viewerのメタファイル↑

}

void ExtendSpeechAreas(vector<Shot> &speechareas){
	vector<Shot> speech_copy(speechareas.size());//サイズを確保して宣言
	std::copy(speechareas.begin(),speechareas.end(),speech_copy.begin());//複製
	for(int i=1;i<speechareas.size()-1;i++){
		if( (speech_copy[i].StartTime - speech_copy[i-1].EndTime) > 2){
			speechareas[i].set_StartTime( speech_copy[i].StartTime -2  );
		}else{
			speechareas[i].set_StartTime(speech_copy[i-1].EndTime);
		}

		if(speech_copy[i+1].StartTime - speech_copy[i].EndTime > 2){
			speechareas[i].set_EndTime(speech_copy[i].EndTime + 2);
		}else{
			speechareas[i].set_EndTime(speech_copy[i+1].StartTime);
		}
	}
}

void CLFlow(char* filename){

	ostringstream moviefile;
	moviefile  << "digestmeta\\" << filename << "\\" << filename << ".mp4";
//	moviefile  << "digestmeta\\" << filename << "\\" << filename << "_test.mp4";//デバッグ用
	// 動画ファイルの読み込み
	VideoCapture capture = VideoCapture(moviefile.str().c_str());
	// TV-L1アルゴリズムによるオプティカルフロー計算オブジェクトの生成
	Ptr<DenseOpticalFlowExt> opticalFlow = superres::createOptFlow_DualTVL1();

	// 前のフレームを保存しておく
	Mat prev,curr;
	capture >> prev;

//	FILE *output_hist;
//	output_hist = fopen("CLFlow_hist.txt", "w");

	FILE *output_sum;
	ostringstream oss_outsum;
	oss_outsum  << "digestmeta\\" << filename << "\\" << filename << "_CLFlow_sum.txt";
	output_sum = fopen(oss_outsum.str().c_str(), "w");

	FILE *output_direction;
	ostringstream oss_direction;
	oss_direction  << "digestmeta\\" << filename << "\\" << filename << "_CLFlow_direction.txt";
	output_direction = fopen(oss_direction.str().c_str(), "w");

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


//		Mat magnitude = Mat::zeros(90, 160, CV_8UC1);

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
				//デバッグ用
//				magnitude.at<unsigned char>(y,x) = 255*sqrt(pow(vx.pData[y*vx.width()+x],2) + pow(vy.pData[y*vx.width()+x],2));
				//デバッグ用

				int k = std::min(999,(int)(100*sqrt(pow(vx.pData[y*vx.width()+x],2) + pow(vy.pData[y*vx.width()+x],2))));
				opt_hist[k]++;
				opt_sum += k;
			}
		}
		fprintf(output_sum,"%d\n",opt_sum);

		fprintf(output_direction,"%lf\n",ClacOpticalFlowMoveDirection(vx,vy));

		//デバッグ用
		
//		ostringstream osstmp;
//		osstmp << "digestmeta\\kihara\\" << count+1 <<".png";
//		cv::imwrite(osstmp.str().c_str(),magnitude);

		//デバッグ用


		/*
		for(int j=0;j<1000;j++){
			fprintf(output_hist, "%d\t",opt_hist[j]);
		}
		fprintf(output_hist, "\n");
		*/

		curr.copyTo(prev);
	}

}

void DetectSpeechArea(char* filename, int volumethreshold, int speechinterval){

	FILE *areafileexisttest;
	std::ostringstream speechareafile;
	speechareafile << "digestmeta\\" << filename << "\\" << filename << "_speechareas_" << volumethreshold << "_" << speechinterval <<  ".txt";
	if ( (areafileexisttest = fopen(speechareafile.str().c_str(),"r")) != NULL ){
		fclose( areafileexisttest );
		// ファイルが存在するので何もしない

	}
	else{
		// ファイルは存在しないので作る

		MakeMonoFile(filename);

		FILE *filelist;
		std::ostringstream filelist_name;
		filelist_name << "digestmeta\\" << filename << "\\" << filename << "_filelist.txt";

		std::ostringstream filelist_contents;
		filelist_contents << "digestmeta\\" << filename << "\\" << filename << "_mono.wav";

		filelist = fopen(filelist_name.str().c_str(), "w");
		fprintf(filelist,filelist_contents.str().c_str());
		fclose(filelist);

		std::ostringstream run_adintool;
		run_adintool << "adintool.exe -in file -filelist " << filelist_name.str().c_str() <<
			" -out file -filename adintool_result\\ -freq 48000 -headmargin " << speechinterval/2 <<
			" -tailmargin " << speechinterval/2 << " -lv " << volumethreshold << " > ";
		run_adintool << "digestmeta\\" << filename << "\\" << filename << "_speechareas_" << volumethreshold << "_" << speechinterval <<".txt";
		int ret = system(run_adintool.str().c_str());
		if (ret != 0){
			printf("Adintool起動失敗\n");
		}
	}
}

double CalcVolumeAveofWav(char* filename){

	int ret;
	std::ostringstream makemonofile,monofilename;
	monofilename << "digestmeta\\" << filename << "\\" << filename << "_mono.wav";

	FILE *fileexisttest;
	if ( (fileexisttest = fopen(monofilename.str().c_str(),"r")) != NULL ){
		fclose( fileexisttest );
		// ファイルが存在する
	}
	else{
		makemonofile << "ffmpeg -i digestmeta\\" << filename << "\\" << filename << ".mp4";
		makemonofile << " -ac 1 digestmeta\\" << filename << "\\" << filename << "_mono.wav";
		ret = system(makemonofile.str().c_str());
		if (ret != 0){
			printf("モノラルファイル作成失敗\n");
		}
	}

	WAVE          wav1;
    WAVE_FORMAT   fmt1;
    vector<short> ch1,ch2;
	double volumeave=0;
        
    //ファイルから読み込み
	ostringstream oss;
	oss << "digestmeta\\" << filename << "\\" << filename << "_mono.wav";
    wav1.load_from_file((char*)oss.str().c_str());
    
    //フォーマット情報とデータを取り出す
    fmt1=wav1.get_format();
    wav1.get_channel(ch1,0);
        
	if(fmt1.num_of_channels == 2){
		for(int i=0;i<ch1.size();i++){
			volumeave += (abs(ch1[i])+abs(ch2[i]));
		}
	}else{
		for(int i=0;i<ch1.size();i++){
			volumeave += abs(ch1[i]);
		}
	}
	return volumeave / (double)ch1.size();
}

bool search_areas(Shot inputshot, double time){
	//timeがinputshotの中にあるかどうか調べる

	if(inputshot.StartTime<time && time<inputshot.EndTime){
		return true;
	}else{
		return false;
	}
}


void AdjustCutAreaBySpeechArea(vector<Shot> &adjustingarea, vector<Shot> &continuityareas, int mode){
	//adjustingarea のstartとendが continuityareas のどこかに入っているかどうか調べて，
	//入っていれば，入っている continuityarea のstart/end まで幅を広げてshotを返す
	//mode １:伸ばす
	//mode ２:縮める

	for(int j=0;j<adjustingarea.size();j++){
		double starttime = adjustingarea[j].StartTime;
		double endtime = adjustingarea[j].EndTime;

//		cout << continuityareas.size() << endl;

		for(int i=0;i<continuityareas.size();i++){
			if(search_areas(continuityareas[i],starttime) == true){
				if(mode == 1){
					adjustingarea[j].set_StartTime(continuityareas[i].StartTime);
				}else{
					if(adjustingarea[j].EndTime < continuityareas[i].EndTime){
						adjustingarea.erase(adjustingarea.begin()+j);
						i--;
					}else{
						adjustingarea[j].set_StartTime(continuityareas[i].EndTime);
					}
				}
				break;
			}
		}
		for(int i=0;i<continuityareas.size();i++){
			if(search_areas(continuityareas[i],endtime) == true){
				if(mode == 1){
					adjustingarea[j].set_EndTime(continuityareas[i].EndTime);
				}else{
					if(adjustingarea[j].StartTime > continuityareas[i].StartTime){
						adjustingarea.erase(adjustingarea.begin()+j);
						i--;
					}else{
						adjustingarea[j].set_EndTime(continuityareas[i].StartTime);
					}
				}
				break;
			}
		}
	}
}

int SearchBoundarySpeech(double time , vector<Shot> &speecharea){
	for(int i=0;i<speecharea.size();i++){
		if(speecharea[i].StartTime <= time && time <= speecharea[i].EndTime){
			return true;
		}
	}
	return false;
}

void AdjustCutAreaBySpeechArea(vector<Shot> &adjustingarea, vector<Shot> &speecharea){
	//speechareaは時間でソートされていることが前提
	//adjustingarea の前後2秒を調べて，最大の切れ目(外側)でカットする
	//切れ目がなかった場合は，2秒伸ばす

	for(int j=0;j<adjustingarea.size();j++){
		double starttime = adjustingarea[j].StartTime;
		double endtime = adjustingarea[j].EndTime;

		//startの方の切る時間を決定
		int start_start_index,start_end_index;
		double start_start_time;

		if(SearchBoundarySpeech(starttime-2,speecharea)==true ){
			if(SearchBoundarySpeech(starttime,speecharea)==true){
				for(int i=0;i<speecharea.size();i++){
					if(speecharea[i].StartTime <= starttime-2 && starttime-2 <= speecharea[i].EndTime){
						start_start_index = i;
					}
					if(speecharea[i].StartTime <= starttime && starttime <= speecharea[i].EndTime){
						start_end_index = i;
					}
				}
				if(start_start_index == start_end_index){
					start_start_time = starttime-2;
				}else{
					vector<sukima> s_sukima;
					for(int i=start_start_index;i<start_end_index;i++){
						sukima tmp;
						tmp.Starttime = speecharea[i].EndTime;
						tmp.Endtime = speecharea[i+1].StartTime;
						tmp.Widthtime = speecharea[i+1].StartTime - speecharea[i].EndTime;
						s_sukima.push_back(tmp);
					}
					std::sort(s_sukima.begin(),s_sukima.end(),sukima::Widthtime_cmp);
					start_start_time = s_sukima[0].Starttime;
				}
			}else{
				bool flag = true;
				for(int i=0;i<speecharea.size();i++){
					if(speecharea[i].StartTime <= starttime-2 && starttime-2 <= speecharea[i].EndTime){
						start_start_index = i;
					}
					if(flag == true && speecharea[i].StartTime > starttime){
						start_end_index = i;
						flag = false;
					}
				}
				vector<sukima> s_sukima;
				for(int i=start_start_index;i<start_end_index;i++){
					sukima tmp;
					tmp.Starttime = speecharea[i].EndTime;
					tmp.Endtime = speecharea[i+1].StartTime;
					tmp.Widthtime = speecharea[i+1].StartTime - speecharea[i].EndTime;
					s_sukima.push_back(tmp);
				}
				std::sort(s_sukima.begin(),s_sukima.end(),sukima::Widthtime_cmp);
				start_start_time = s_sukima[0].Starttime;
			}
		}else{
			if(SearchBoundarySpeech(starttime,speecharea)==true){
				bool flag = true;
				for(int i=0;i<speecharea.size();i++){
					if(flag == true && speecharea[i].StartTime > starttime-2){
						start_start_index = i;
						flag = false;
					}
					if(speecharea[i].StartTime <= starttime && starttime <= speecharea[i].EndTime){
						start_end_index = i;
					}
				}
				vector<sukima> s_sukima;
				sukima tmp2;
				tmp2.Starttime = starttime-2;
				tmp2.Endtime = speecharea[start_start_index].StartTime;
				tmp2.Widthtime = speecharea[start_start_index].StartTime - (starttime-2);
				s_sukima.push_back(tmp2);

				for(int i=start_start_index;i<start_end_index;i++){
					sukima tmp;
					tmp.Starttime = speecharea[i].EndTime;
					tmp.Endtime = speecharea[i+1].StartTime;
					tmp.Widthtime = speecharea[i+1].StartTime - speecharea[i].EndTime;
					s_sukima.push_back(tmp);
				}
				std::sort(s_sukima.begin(),s_sukima.end(),sukima::Widthtime_cmp);
				start_start_time = s_sukima[0].Starttime;

			}else{
				bool flag1 = true;
				bool flag2 = true;
				for(int i=0;i<speecharea.size();i++){
					if(flag1 == true && speecharea[i].StartTime > starttime-2){
						start_start_index = i;
						flag1 = false;
					}
					if(flag2 == true && speecharea[i].StartTime > starttime){
						start_end_index = i;
						flag2 = false;
					}
				}
				if(start_start_index == start_end_index){
					start_start_time = starttime -2;
				}else{
					vector<sukima> s_sukima;
					sukima tmp2;
					tmp2.Starttime = starttime-2;
					tmp2.Endtime = speecharea[start_start_index].StartTime;
					tmp2.Widthtime = speecharea[start_start_index].StartTime - (starttime-2);
					s_sukima.push_back(tmp2);

					for(int i=start_start_index;i<start_end_index;i++){
						sukima tmp;
						tmp.Starttime = speecharea[i].EndTime;
						tmp.Endtime = speecharea[i+1].StartTime;
						tmp.Widthtime = speecharea[i+1].StartTime - speecharea[i].EndTime;
						s_sukima.push_back(tmp);
					}
					std::sort(s_sukima.begin(),s_sukima.end(),sukima::Widthtime_cmp);
					start_start_time = s_sukima[0].Starttime;
				}
			}
		}


		//endの方の切る時間を決定
		int end_start_index,end_end_index;
		double end_end_time;

		if(SearchBoundarySpeech(endtime,speecharea)==true ){
			if(SearchBoundarySpeech(endtime+2,speecharea)==true){

				for(int i=0;i<speecharea.size();i++){
					if(speecharea[i].StartTime <= endtime && endtime <= speecharea[i].EndTime){
						end_start_index = i;
					}
					if( ( speecharea[i].StartTime <= (endtime+2) ) &&  ( (endtime+2) <= speecharea[i].EndTime ) ){
						end_end_index = i;
					}
				}
				if(end_start_index == end_end_index){
					end_end_time = endtime+2;
				}else{
					vector<sukima> s_sukima;
					for(int i=end_start_index;i<end_end_index;i++){
						sukima tmp;
						tmp.Starttime = speecharea[i].EndTime;
						tmp.Endtime = speecharea[i+1].StartTime;
						tmp.Widthtime = speecharea[i+1].StartTime - speecharea[i].EndTime;
						s_sukima.push_back(tmp);
					}
					std::sort(s_sukima.begin(),s_sukima.end(),sukima::Widthtime_cmp);
					end_end_time = s_sukima[0].Endtime;
				}
			}else{
				bool flag = true;
				for(int i=0;i<speecharea.size();i++){
					if(flag == true && speecharea[i].StartTime > endtime + 2){
						end_end_index = i-1;
						flag = false;
					}
					if( ( speecharea[i].StartTime <= (endtime)  ) && ( (endtime) <= speecharea[i].EndTime ) ){
						end_start_index = i;
					}
				}
				vector<sukima> s_sukima;
				for(int i=end_start_index;i<end_end_index;i++){
					sukima tmp;
					tmp.Starttime = speecharea[i].EndTime;
					tmp.Endtime = speecharea[i+1].StartTime;
					tmp.Widthtime = speecharea[i+1].StartTime - speecharea[i].EndTime;
					s_sukima.push_back(tmp);
				}

				sukima tmp2;
				tmp2.Starttime = speecharea[end_end_index].StartTime;
				tmp2.Endtime = endtime+2;
				tmp2.Widthtime = endtime+2 - speecharea[end_end_index].StartTime;
				s_sukima.push_back(tmp2);

				std::sort(s_sukima.begin(),s_sukima.end(),sukima::Widthtime_cmp);
				end_end_time = s_sukima[0].Endtime;
			}
		}else{
			if(SearchBoundarySpeech(endtime+2,speecharea)==true){
				bool flag = true;
				for(int i=0;i<speecharea.size();i++){
					if(speecharea[i].StartTime <= endtime+2 && endtime+2 <= speecharea[i].EndTime){
						end_end_index = i;
					}
					if(flag == true && ( speecharea[i].StartTime > (endtime) ) ){
						end_start_index = i-1;
						flag = false;
					}
				}
				vector<sukima> s_sukima;
				for(int i=end_start_index;i<end_end_index;i++){
					sukima tmp;
					tmp.Starttime = speecharea[i].EndTime;
					tmp.Endtime = speecharea[i+1].StartTime;
					tmp.Widthtime = speecharea[i+1].StartTime - speecharea[i].EndTime;
					s_sukima.push_back(tmp);
				}
				std::sort(s_sukima.begin(),s_sukima.end(),sukima::Widthtime_cmp);
				end_end_time = s_sukima[0].Endtime;
			}else{
				bool flag1 = true;
				bool flag2 = true;
				for(int i=0;i<speecharea.size();i++){
					if(flag1 == true && speecharea[i].StartTime > endtime){
						end_start_index = i-1;
						flag1 = false;
					}
					if(flag2 == true && speecharea[i].StartTime > endtime+2){
						end_end_index = i-1;
						flag2 = false;
					}
				}
				if(end_start_index == end_end_index){
					end_end_time = endtime+2;
				}else{
					vector<sukima> s_sukima;
					for(int i=end_start_index;i<end_end_index;i++){
						sukima tmp;
						tmp.Starttime = speecharea[i].EndTime;
						tmp.Endtime = speecharea[i+1].StartTime;
						tmp.Widthtime = speecharea[i+1].StartTime - speecharea[i].EndTime;
						s_sukima.push_back(tmp);
					}
					sukima tmp2;
					tmp2.Starttime = speecharea[end_end_index].StartTime;
					tmp2.Endtime = endtime+2;
					tmp2.Widthtime = endtime+2 - speecharea[end_end_index].StartTime;
					s_sukima.push_back(tmp2);

					std::sort(s_sukima.begin(),s_sukima.end(),sukima::Widthtime_cmp);
					end_end_time = s_sukima[0].Endtime;
				}
			}
		}
		adjustingarea[j].StartTime = start_start_time;
		adjustingarea[j].EndTime = end_end_time;
	}
}

vector<Shot> SpeechBasedCut(char* filename, int volumethreshold, int speechinterval){
	vector<Shot> out;
	Shot tmp;
	
	ostringstream oss;
	oss << "digestmeta\\" << filename << "\\" << filename << "_speechareas_" << volumethreshold << "_" << speechinterval <<".txt";

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

void JointSpeechAreas(vector<Shot> &speechareas, double speechinterval,double maxtime){

	bool flag = false;
	int joint_start, joint_end;

	for(int i=0;i<speechareas.size()-1;i++){
		if((speechareas[i+1].StartTime - speechareas[i].EndTime) < speechinterval){
			if(flag == false){
				joint_start = i;
				flag = true;
			}
		}else{
			if(flag == true){
				joint_end = i;
				vector<Shot> CuttenAreas;
				CuttenAreas = CutSpeechArea(speechareas,maxtime,joint_start,joint_end);
				flag = false;
				//speechareas のjoint_startからjoint_endまでを消して，CuttenAreasを追加する
				speechareas.erase(speechareas.begin() + joint_start,speechareas.begin() + joint_end + 1);
				speechareas.insert(speechareas.begin() + joint_start, CuttenAreas.begin(), CuttenAreas.end());

				i=joint_start + CuttenAreas.size() -1 ;
			}
		}
	}
}

vector<Shot> CutSpeechArea(vector<Shot> &speechareas,double maxtime,int joint_start,int joint_end){
	vector<Shot> out;
	if( speechareas[joint_end].EndTime - speechareas[joint_start].StartTime < maxtime){
		Shot tmp;
		tmp.StartTime = speechareas[joint_start].StartTime;
		tmp.EndTime = speechareas[joint_end].EndTime;
		out.push_back(tmp);
	}else{
		int MaxInterval_start;
		double MaxInterval=0;
		if(joint_end - joint_start <= 1){
			out.insert(out.end(), speechareas.begin() + joint_start,speechareas.begin() + joint_end + 1);
		}else{
			for(int i=joint_start;i<joint_end;i++){
				if(i==joint_start){
					MaxInterval = speechareas[i+1].StartTime - speechareas[i].EndTime;
					MaxInterval_start = i;
				}else if(speechareas[i+1].StartTime - speechareas[i].EndTime > MaxInterval){
					MaxInterval = speechareas[i+1].StartTime - speechareas[i].EndTime;
					MaxInterval_start = i;
				}
			}
			//一番大きいintervalで分けたときの，前半分
			vector<Shot> FirstHalf = CutSpeechArea(speechareas,maxtime,joint_start,MaxInterval_start);
			out.insert(out.end(), FirstHalf.begin(),FirstHalf.end());

			//一番大きいintervalで分けたときの，後半分
			vector<Shot> SecondHalf = CutSpeechArea(speechareas,maxtime,MaxInterval_start+1,joint_end);
			out.insert(out.end(), SecondHalf.begin(),SecondHalf.end());
		}
	}
	return out;
}

vector<Shot> MakeOpticalFlowBasedUncuttableAreaVector(char* filename){
	vector<Shot> out;
	Shot tmp;
	double tmp_starttime,tmp_endtime;

	ostringstream oss;
	oss << "digestmeta\\" << filename << "\\" << filename << "_CLFlow_sum.txt";
	std::ifstream ifs(oss.str().c_str());
    if (ifs.fail())
    {
		CLFlow(filename);
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
	const double THRE =  OpticalFlowUnCuttableRate * average;

	bool flag = false;
	for(int i=0;i<MAveData.size();i++){
		if(flag == true){
			if(MAveData[i] > THRE){
				tmp.EndTime = (double)i/FPS;
				out.push_back(tmp);

				flag = false;
			}
		}else if(MAveData[i] < THRE){
			tmp.StartTime = (double)i/FPS;

			flag = true;
		}
	}

	return out;
}

vector<short> readwave(char* filename)
{
    WAVE          wav1;
    WAVE_FORMAT   fmt1;
    vector<short> ch1,ch2,chMix;
        
    //ファイルから読み込み
	ostringstream oss;
	oss << "digestmeta\\" << filename << "\\" << filename << "_mono.wav";
    wav1.load_from_file((char*)oss.str().c_str());
    
    //フォーマット情報とデータを取り出す
    fmt1=wav1.get_format();
    wav1.get_channel(ch1,0);
    
    //フォーマット情報表示
	/*
    cout<<endl;
    cout<<"format id       = "<<fmt1.format_id      <<endl;
    cout<<"channels        = "<<fmt1.num_of_channels<<"\t[Ch]"<<endl;
    cout<<"sampling rate   = "<<fmt1.samples_per_sec<<"\t[Hz]"<<endl;
    cout<<"bytes per sec   = "<<fmt1.bytes_per_sec  <<"\t[bytes/sec]"<<endl;
    cout<<"block size      = "<<fmt1.block_size     <<"\t[bytes]"<<endl;
    cout<<"bits per sample = "<<fmt1.bits_per_sample<<"\t[bits/sample]"<<endl;
    cout<<endl;
	*/

	double max=0,min=0;
    
	if(fmt1.num_of_channels == 2){
		for(int i=0;i<ch1.size();i++){
			chMix.push_back(ch1[i]+ch2[i]);
		}
	}else{
		for(int i=0;i<ch1.size();i++){
			chMix.push_back(ch1[i]);
		}
	}
	return chMix;
}

int samplingrateofwave(char* filename)
{
    WAVE          wav1;
    WAVE_FORMAT   fmt1;
    vector<short> ch1,ch2,chMix;
        
    //ファイルから読み込み
	ostringstream oss;
	oss << "digestmeta\\" << filename << "\\" << filename << "_mono.wav";
    wav1.load_from_file((char*)oss.str().c_str());
    
    //フォーマット情報とデータを取り出す
    fmt1=wav1.get_format();
    wav1.get_channel(ch1,0);
    
	return fmt1.samples_per_sec;
}


vector<double> CalcMoveAve(vector<double> inputvector, int Range){
	vector<double> MoveAveResult;
	for(int i=0;i<inputvector.size();i++){
		double tmp=0;
		double counter=0;
		for(int j = std::min(i-Range,0) ; j<i+Range && j<inputvector.size() ;j++){
			tmp += inputvector[j];
			counter++;
		}
		MoveAveResult.push_back(tmp/counter);
	}
	return MoveAveResult;
}

vector<double> CalcMoveAve(vector<short> inputvector, int Range){
	vector<double> MoveAveResult;
	for(int i=0;i<inputvector.size();i++){
		double tmp=0;
		double counter=0;
		for(int j = std::min(i-Range,0) ; j<i+Range && j<inputvector.size() ;j++){
			tmp += inputvector[j];
			counter++;
		}
		MoveAveResult.push_back(tmp/counter);
	}
	return MoveAveResult;
}

vector<double> CalcAve(vector<short> inputvector, int Range){
	vector<double> AveResult;
	double tmp=0;
	double counter=0;
	for(int i=0;i<inputvector.size();i++){
		tmp += abs(inputvector[i]);
		counter++;
		if( (i+1)%Range == 0 || i == inputvector.size()-1 ){
			AveResult.push_back(tmp / counter);
			tmp = 0;
			counter = 0;
		}
	}
	return AveResult;
}

double ClacOpticalFlowMoveDirection(DImage vx, DImage vy){
	double S = 0, Tx = 0, Ty = 0;
	for(int y = 0; y < vx.height(); ++y){
		for(int x = 0; x < vx.width(); ++x){
			S += sqrt(pow(vx.pData[y*vx.width()+x],2) + pow(vy.pData[y*vx.width()+x],2));
			Tx += vx.pData[y*vx.width()+x];
			Ty += vy.pData[y*vx.width()+x];
		}
	}
	return sqrt(pow(Tx,2) + pow(Ty,2)) / S;
}

void CalcFlowAve(vector<Shot> &shots, char* filename){
	ostringstream oss;
	oss << "digestmeta\\" << filename << "\\" << filename << "_CLFlow_sum.txt";
	std::ifstream ifs(oss.str().c_str());
	if (ifs.fail())
	{
		CLFlow(filename);
	}

	std::vector<double> RawData;
	std::copy(std::istream_iterator<double>(ifs), std::istream_iterator<double>(), std::back_inserter(RawData));

	for(int j=0;j<shots.size();j++){
		double tmp=0;
		for(int i=shots[j].StartTime*FPS;i<shots[j].EndTime*FPS;i++){
			tmp += RawData[i];
		}
		shots[j].set_FlowAverage(tmp / ( ( shots[j].EndTime - shots[j].StartTime ) * FPS ) );
	}
}

void CalcVolumeChangeAveandSum(vector<Shot> &shots, char* filename, double threshold){

	vector<short> videovolume = readwave(filename);
	const int SamplingRate = samplingrateofwave(filename);
	vector<double> videovolume_ave = CalcAve(videovolume,SamplingRate*VolumeAveRange);

	for(int j=0;j<shots.size();j++){
		double volume = 0;
		for(int i = (shots[j].StartTime / VolumeAveRange );i<(shots[j].EndTime / VolumeAveRange)-1 ; i++){
			if(double tmp = abs( videovolume_ave[i] - videovolume_ave[i+1] ) < threshold)
			volume += tmp ; 
		}
		shots[j].set_VolumeAverage( volume / (shots[j].EndTime - shots[j].StartTime) * VolumeAveRange );
		shots[j].set_VolumeChangeSum(volume);
	}
}

void CalcVolumeChangeAveandSum(vector<Shot> &shots, char* filename){

	vector<short> videovolume = readwave(filename);
	const int SamplingRate = samplingrateofwave(filename);
	vector<double> videovolume_ave = CalcAve(videovolume,SamplingRate*VolumeAveRange);

	for(int j=0;j<shots.size();j++){
		double volume = 0;
		double volumeUnderThre = 0;
		double volumeOverThre = 0;
		for(int i = (shots[j].StartTime / VolumeAveRange );i<(shots[j].EndTime / VolumeAveRange)-1 ; i++){
			volume += abs( videovolume_ave[i] - videovolume_ave[i+1] ) ; 
			if(double tmp1 = abs( videovolume_ave[i] - videovolume_ave[i+1] ) > VolumeThre){
				volumeOverThre += tmp1 ; 
			}
			if(double tmp2 = abs( videovolume_ave[i] - videovolume_ave[i+1] ) < VolumeThre){
				volumeUnderThre += tmp2 ; 
			}
		}
		shots[j].set_VolumeAverage( volume / (shots[j].EndTime - shots[j].StartTime) * VolumeAveRange );
		shots[j].set_VolumeChangeThreOverSum(volume);
//		shots[j].set_VolumeChangeThreOverSum(volumeOverThre);
		shots[j].set_VolumeChangeThreUnderSum(volumeUnderThre);
	}
}


void CalcFlowDirection(vector<Shot> &shots, char* filename){

	ostringstream oss;
	oss << "digestmeta\\" << filename << "\\" << filename << "_CLFlow_direction.txt";
	std::ifstream ifs(oss.str().c_str());
	if (ifs.fail())
	{
		CLFlow(filename);
	}

	std::vector<double> RawData;
	std::copy(std::istream_iterator<double>(ifs), std::istream_iterator<double>(), std::back_inserter(RawData));

	for(int j=0;j<shots.size();j++){
		double tmp=0;
		for(int i=shots[j].StartTime*FPS;i<shots[j].EndTime*FPS;i++){
			tmp += RawData[i];
		}
		shots[j].set_FlowDirection(tmp / ( ( shots[j].EndTime - shots[j].StartTime ) * FPS ) );
	}
}

void CalcVolumeAndFlowMixPriority(vector<Shot> &shots, double volume_ratio, double optflow_ratio){
	for(int i=0;i<shots.size();i++){
		shots[i].set_VolumeAndFlowMixPriority(volume_ratio * shots[i].VolumeChangeSum - optflow_ratio * shots[i].FlowAverage);
	}
}


void CalcVolumeAndFlowMixPriority(vector<Shot> &shots, double volume_ratio,double volume_thre_ratio , double optflow_ratio){
	for(int i=0;i<shots.size();i++){
		shots[i].set_VolumeAndFlowMixPriority(volume_ratio * (shots[i].VolumeChangeThreOverSum)/324  + volume_thre_ratio * (shots[i].VolumeChangeThreUnderSum) - optflow_ratio * shots[i].FlowAverage/2069);
	}
}


























//以下，OpenCVのデフォルト関数を用いたオプティカルフローの計算
//なぜかピークが現れた
void OpenCVDefaultFlow(char *filename){

	ostringstream moviefile;
	moviefile  << "digestmeta\\" << filename << "\\" << filename << "_test.mp4";
	// 動画ファイルの読み込み
	VideoCapture capture = VideoCapture(moviefile.str().c_str());
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

		Mat magnitude, angle;
		cartToPolar(flowX, flowY, magnitude, angle, true);

		Mat mat_ir;
		magnitude.convertTo(mat_ir, CV_8UC1, 255.0);

		ostringstream osstmp;
		osstmp << "digestmeta\\kihara\\" << count+1 <<".png";
		cv::imwrite(osstmp.str().c_str(),mat_ir);

		int tmp = 0;
		for(int y=0;y<magnitude.rows;y++){
			for(int x=0;x<magnitude.cols;x++){
				tmp += magnitude.at<unsigned char>(y,x);
			}
		}
		cout << tmp ;

		///////////↓デバッグ用↓
		/*

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

//		imshow("magnitude",magnitude);
//		cvWaitKey(0);

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

		printf(" \nflow_sum : %lf\n", flow_sum );
		fprintf(output_sum ," %lf\n", flow_sum );

		for(int j=0;j<400;j++){
			fprintf(output_hist, "%d\t",opt_hist[j] );
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

/*
int OpticalFlowBasedCutandAdjustShots(char* filename){

	//ベース
	//オプティカルフローをベースにショットを作成
	vector<Shot> OpticalFlowMinArea = MakeShotBaseByOpticalFlow(filename);

	//speechareaを作る
	vector<Shot> speecharea = MakeShotBaseBySpeech(filename,2,MaxTime);

	//speechareaから，発話の途中でショットがカットされないように調整
	AdjustCutAreaBySpeechArea(OpticalFlowMinArea,speecharea,2);

	//flowの１５秒の平均が小さい順にソート
	//（FlowMinがもともと１５秒平均の値であることに注意）
	std::sort(OpticalFlowMinArea.begin(), OpticalFlowMinArea.end(), &Shot::flow_cmp);

	//結果をファイルに保存
	FILE *output_optbasecut;
	ostringstream oss_output_optbasecut;
	oss_output_optbasecut  << "digestmeta\\" << filename << "\\" << filename << "_optical_flow_base_cut.txt";
	output_optbasecut = fopen(oss_output_optbasecut.str().c_str(), "w");

	fprintf(output_optbasecut,"\t\n");

	for(int i=0;i<10;i++){
		ostringstream tmp;
		tmp << OpticalFlowMinArea[i].StartTime << "\t" << OpticalFlowMinArea[i].EndTime << "\t" << endl; 
//			(int)OpticalFlowMinArea[i].StartTime/60  << " : " << (int)OpticalFlowMinArea[i].StartTime%60 <<
//			" - " << (int)OpticalFlowMinArea[i].EndTime/60 << " : " << (int)OpticalFlowMinArea[i].EndTime%60 << endl;

		fprintf(output_optbasecut,tmp.str().c_str());
	}
    return 0;
}
*/
/*
void tmp(){
	for(int i=0;i<10;i++){
		Mat x,y;
		ostringstream osstmp1,osstmp2;
		osstmp1 << "digestmeta\\kihara\\x_" << i+1 <<".jpg";
		osstmp2 << "digestmeta\\kihara\\y_" << i+1 <<".jpg";
		x =imread(osstmp1.str().c_str());
		y =imread(osstmp2.str().c_str());

		Mat magnitude, angle;
		cartToPolar(x, y, magnitude, angle, true);

		Mat mat_ir;
		magnitude.convertTo(mat_ir, CV_8UC1, 255.0);

		ostringstream osstmp;
		osstmp << "digestmeta\\kihara\\" << i+1 <<".png";
		cv::imwrite(osstmp.str().c_str(),mat_ir);
	}
}
*/
/*
int SoundBasedAreaCut(char* filename){

	//ベース
	//発話区間をadintoolで計算し， speecharea に入れる
	vector<Shot> speecharea = MakeShotBaseBySpeech(filename,2,15);

	//調整
	//7秒に満たないものは除外する
	for(int i=0;i<speecharea.size();){
		if( (speecharea[i].EndTime - speecharea[i].StartTime) < 7){
			speecharea.erase(speecharea.begin()+i);
		}else{
			i++;
		}
	}

	//CLFlow_sumから，FlowAverageを計算
	CalcFlowAve(speecharea,filename);

	//FlowAverageを基に，あまりに動きが大きいところは区間から除外するようにする
	for(auto shot = speecharea.begin() ; shot !=speecharea.end() ; ){
		if(shot->FlowAverage > FLOWTHRE){
			speecharea.erase(shot);
		}else{
			shot++;
		}
	}


	//ソート
	//VolumeAverageを計算
	CalcVolumeChangeAveandSum(speecharea,filename);

	//VolumeChangeSumを基に，大きい順にソート
	std::sort(speecharea.begin(), speecharea.end(), &Shot::volumeChangeSum_cmp);

	//結果をファイルに保存
	FILE *output_soundbasecut;
	ostringstream oss_output_soundbasecut;
	oss_output_soundbasecut  << "digestmeta\\" << filename << "\\" << filename << "_sound_base_cut.txt";
	output_soundbasecut = fopen(oss_output_soundbasecut.str().c_str(), "w");

	fprintf(output_soundbasecut,"\t\n");
	for(int i=0;i<10;i++){
		ostringstream tmp;
		tmp << speecharea[i].StartTime << "\t" << speecharea[i].EndTime << "\t" << endl ;
//		<< (int)speecharea[i].StartTime/60  << " : " << (int)speecharea[i].StartTime%60 <<
//			" - " << (int)speecharea[i].EndTime/60 << " : " << (int)speecharea[i].EndTime%60 << endl;

		fprintf(output_soundbasecut,tmp.str().c_str());
	}

	return 0;
}

void CutAndDefinePriority(char* filename, int Base, double volume_ratio, double optflow_ratio, int adjustmode){
	//Baseが1なら，speech
	//Baseが2なら，OpticalFlow


	//ベース作り
	vector<Shot> shots;
	if(Base == 1){
		cout << "Speech Base Cut Start!!" << endl;
		cout << "ベース作り開始" << endl;
		shots = MakeShotBaseBySpeech(filename,2,MaxTime);
	}else{
		cout << "Optical Flow Base Cut Start!!" << endl;
		cout << "ベース作り開始" << endl;
		shots = MakeShotBaseByOpticalFlow(filename);
	}

	//範囲の調整
	cout << "範囲調整開始" << endl;
	if(Base == 1){
		//7秒に満たないものは除外する
		for(int i=0;i<shots.size();){
			if( (shots[i].EndTime - shots[i].StartTime) < 7){
				shots.erase(shots.begin()+i);
			}else{
				i++;
			}
		}
		//CLFlow_sumから，FlowAverageを計算
		CalcFlowAve(shots,filename);

		//CLFlowの全体の平均と分散を計算
		ostringstream oss;
		oss << "digestmeta\\" << filename << "\\" << filename << "_CLFlow_sum.txt";
		std::ifstream ifs(oss.str().c_str());

		//CLFlowの結果をRawDataにいったん格納し，その後移動平均を取ってMAveDataに格納
		std::vector<double> RawData;
		std::vector<double> MAveData;
		std::copy(std::istream_iterator<double>(ifs), std::istream_iterator<double>(), std::back_inserter(RawData));

		double average=0, dispersion=0;
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

		for(int i=0;i<MAveData.size();i++){
			dispersion += pow( MAveData[i] - average , 2) ;
		}
		dispersion = sqrt(dispersion / MAveData.size() );

		const double THRE = average + dispersion;//MAveDataの平均から１σを閾値と定義		

		//FlowAverageを基に，あまりに動きが大きいところは区間から除外するようにする
		for(auto shot = shots.begin() ; shot !=shots.end() ; ){
			if(shot->FlowAverage > THRE){
				shots.erase(shot);
			}else{
				shot++;
			}
		}
		
		//被っている時間が半分以上ある場合は，VolumeChangeAveandSumの値が大きい方を残して，小さい方を消す
		CalcVolumeChangeAveandSum(shots,filename);	//ショットのVolumeChangeSumを計算 .VolumeChangeSum
		for(int i=0;i<shots.size()-1;i++){
			if((shots[i].StartTime + (shots[i].EndTime - shots[i].StartTime)/2 ) > shots[i+1].StartTime){
				if(shots[i].VolumeChangeSum > shots[i+1].VolumeChangeSum){
					shots.erase(shots.begin()+i+1);
				}else{
					shots.erase(shots.begin()+i);
				}
			}
		}

	}else{
		//speechareaを作る
		vector<Shot> speecharea = MakeShotBaseBySpeech(filename,2,MaxTime);

		cout << shots.size() << endl;
		//speechareaから，発話の途中でショットがカットされないように調整
		AdjustCutAreaBySpeechArea(shots,speecharea,adjustmode);

		//7秒に満たないものは除外する
		for(int i=0;i<shots.size();){
			if( (shots[i].EndTime - shots[i].StartTime) < 7){
				shots.erase(shots.begin()+i);
			}else{
				i++;
			}
		}
		cout << shots.size() << endl;

		//被っている時間が半分以上ある場合は，FlowAveの値が小さい方を残して，大きい方を消す
		CalcFlowAve(shots,filename);
		for(int i=0;i<shots.size()-1;){
			if((shots[i].StartTime + (shots[i].EndTime - shots[i].StartTime)/2 ) > shots[i+1].StartTime){
				if(shots[i].FlowAverage < shots[i+1].FlowAverage){
					shots.erase(shots.begin()+i+1);
				}else{
					shots.erase(shots.begin()+i);
				}
			}else{
				i++;
			}
		}

	}
	//ソート
	cout << "ソート開始" << endl;
	CalcFlowAve(shots,filename);				//ショットのオプティカルフローの平均を計算 .FlowAve
	CalcVolumeChangeAveandSum(shots,filename);	//ショットのVolumeChangeSumを計算 .VolumeChangeSum
	CalcFlowDirection(shots,filename);			//ショットのFlowDirectionの平均を計算（使わなさそう）
	CalcVolumeAndFlowMixPriority(shots,volume_ratio,optflow_ratio);					//volume_ratio,optflow_ratioの比で優先度を定義
	std::sort(shots.begin(), shots.end(), &Shot::VolumeAndFlowMixPriority_cmp);		//実際にソートする関数
//	std::sort(shots.begin(), shots.end(), &Shot::VolumeChangeThreSum_cmp);	

	//結果をファイルに保存
	cout << "ファイル出力開始" << endl;
	FILE *output;
	ostringstream oss_output;
	if(Base == 1){
		oss_output  << "digestmeta\\" << filename << "\\" << filename << "_sound_base_cut_" << volume_ratio << "_" << optflow_ratio << "_" << adjustmode<<".txt";
		output = fopen(oss_output.str().c_str(), "w");
	}else{
		oss_output  << "digestmeta\\" << filename << "\\" << filename << "_optflow_base_cut_" << volume_ratio << "_" << optflow_ratio << "_" << adjustmode<<".txt";
		output = fopen(oss_output.str().c_str(), "w");
	}

	fprintf(output,"\t\n");
	for(int i=0;i<shots.size();i++){
		ostringstream tmp;
		tmp << shots[i].StartTime << "\t" << shots[i].EndTime << "\t" << endl ;

		fprintf(output,tmp.str().c_str());
	}
}

void CutAndDefinePriority(char* filename){
	//shot[1]：speech
	//shot[2]：OpticalFlow

	//ベース作り
	cout << "ベース作り開始" << endl;

	vector<Shot> shots[2];
	shots[0] = MakeShotBaseBySpeech(filename,2,11); //真ん中を11秒にして，最大でそれを２秒ずつ両端に伸ばしたMAX１５秒の範囲をとってくる
	shots[1] = MakeShotBaseByOpticalFlow(filename);

	//範囲の調整
	cout << "範囲調整開始" << endl;


	//発話
	//7秒に満たないものは除外する
	for(int i=0;i<shots[0].size();){
		if( (shots[0][i].EndTime - shots[0][i].StartTime) < 7){
			shots[0].erase(shots[0].begin()+i);
		}else{
			i++;
		}
	}
	//CLFlow_sumから，FlowAverageを計算
	CalcFlowAve(shots[0],filename);

	//CLFlowの全体の平均と分散を計算
	ostringstream oss;
	oss << "digestmeta\\" << filename << "\\" << filename << "_CLFlow_sum.txt";
	std::ifstream ifs(oss.str().c_str());

	//CLFlowの結果をRawDataにいったん格納し，その後移動平均を取ってMAveDataに格納
	std::vector<double> RawData;
	std::vector<double> MAveData;
	std::copy(std::istream_iterator<double>(ifs), std::istream_iterator<double>(), std::back_inserter(RawData));

	double average=0, dispersion=0;
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

	for(int i=0;i<MAveData.size();i++){
		dispersion += pow( MAveData[i] - average , 2) ;
	}
	dispersion = sqrt(dispersion / MAveData.size() );

	const double THRE = average + dispersion;//MAveDataの平均から１σを閾値と定義		

	//FlowAverageを基に，あまりに動きが大きいところは区間から除外するようにする
	for(auto shot = shots[0].begin() ; shot !=shots[0].end() ; ){
		if(shot->FlowAverage > THRE){
			shots[0].erase(shot);
		}else{
			shot++;
		}
	}

	//被っている時間が半分以上ある場合は，VolumeChangeAveandSumの値が大きい方を残して，小さい方を消す
	CalcVolumeChangeAveandSum(shots[0],filename);	//ショットのVolumeChangeSumを計算 .VolumeChangeSum
	for(int i=0;i<shots[0].size()-1;i++){
		if((shots[0][i].StartTime + (shots[0][i].EndTime - shots[0][i].StartTime)/2 ) > shots[0][i+1].StartTime){
			if(shots[0][i].VolumeChangeSum > shots[0][i+1].VolumeChangeSum){
				shots[0].erase(shots[0].begin()+i+1);
			}else{
				shots[0].erase(shots[0].begin()+i);
			}
		}
	}



	//オプティカルフロー
	//speechareaを作る
	vector<Shot> speecharea = MakeShotBaseBySpeech(filename,2,MaxTime);

	cout << shots[1].size() << endl;
	//speechareaから，発話の途中でショットがカットされないように調整
	AdjustCutAreaBySpeechArea(shots[1],speecharea,1);

	//7秒に満たないものは除外する
	for(int i=0;i<shots[1].size();){
		if( (shots[1][i].EndTime - shots[1][i].StartTime) < 7){
			shots[1].erase(shots[1].begin()+i);
		}else{
			i++;
		}
	}

	//被っている時間が半分以上ある場合は，FlowAveの値が小さい方を残して，大きい方を消す
	CalcFlowAve(shots[1],filename);
	for(int i=0;i<shots[1].size()-1;){
		if((shots[1][i].StartTime + (shots[1][i].EndTime - shots[1][i].StartTime)/2 ) > shots[1][i+1].StartTime){
			if(shots[1][i].FlowAverage < shots[1][i+1].FlowAverage){
				shots[1].erase(shots[1].begin()+i+1);
			}else{
				shots[1].erase(shots[1].begin()+i);
			}
		}else{
			i++;
		}
	}

	//ソート
	cout << "ソート開始" << endl;
	CalcFlowAve(shots[1],filename);				//ショットのオプティカルフローの平均を計算 .FlowAve
	CalcVolumeChangeAveandSum(shots[1],filename);	//ショットのVolumeChangeSumを計算 .VolumeChangeSum
	CalcFlowDirection(shots[1],filename);			//ショットのFlowDirectionの平均を計算（使わなさそう）

	cout << "ファイル出力開始" << endl;

	vector<Shot> all_result_time_sort;


	////デバッグ
	FILE *output_debug;
	ostringstream oss_debug;
	oss_debug << "digestmeta\\" << filename << "\\" << filename << "_debug.txt";
	output_debug = fopen(oss_debug.str().c_str(), "w");
	for(int i=0;i<shots[0].size();i++){
		fprintf(output_debug ,"%lf\t%lf\t%lf\n",shots[0][i].FlowAverage,shots[0][i].VolumeChangeSum,shots[0][i].VolumeChangeThreSum);
	}
	for(int i=0;i<shots[1].size();i++){
		fprintf(output_debug ,"%lf\t%lf\t%lf\n",shots[0][i].FlowAverage,shots[0][i].VolumeChangeSum,shots[0][i].VolumeChangeThreSum);
	}
	////デバッグ


	//結果をファイルに保存
	FILE *output[2][3];
	FILE *output_all,*output_alldata;

	ostringstream oss_output[2][3];
	ostringstream oss_all,oss_alldata;

	oss_output[0][0]  << "digestmeta\\" << filename << "\\" << filename << "_sound_cut_";
	oss_output[0][1]  << "digestmeta\\" << filename << "\\" << filename << "_sound_cut_";
	oss_output[0][2]  << "digestmeta\\" << filename << "\\" << filename << "_sound_cut_";

	oss_output[1][0]  << "digestmeta\\" << filename << "\\" << filename << "_opt_cut_";
	oss_output[1][1]  << "digestmeta\\" << filename << "\\" << filename << "_opt_cut_";
	oss_output[1][2]  << "digestmeta\\" << filename << "\\" << filename << "_opt_cut_";

	oss_all  << "digestmeta\\" << filename << "\\" << filename << "_all_cut_all_sort.txt";
	output_all = fopen(oss_all.str().c_str(), "w");

	oss_alldata  << "digestmeta\\" << filename << "\\" << filename << "_alldata.txt";
	output_alldata = fopen(oss_alldata.str().c_str(), "w");
	fprintf(output_alldata ,"オプティカルフロー\t音量積分\t音量閾値有積分\n");

	for(int j=0;j<2;j++){
		oss_output[j][0]  << "opt_sort.txt";
		oss_output[j][1]  << "sound_sort.txt";
		oss_output[j][2]  << "sound_thre_sort.txt";

		output[j][0] = fopen(oss_output[j][0].str().c_str(), "w");
		output[j][1] = fopen(oss_output[j][1].str().c_str(), "w");
		output[j][2] = fopen(oss_output[j][2].str().c_str(), "w");

		vector<Shot> Best5s;

		//オプティカルフローソート
		CalcVolumeAndFlowMixPriority(shots[j],0,1);										//volume_ratio,optflow_ratioの比で優先度を定義
		std::sort(shots[j].begin(), shots[j].end(), &Shot::VolumeAndFlowMixPriority_cmp);		//実際にソートする関数
		for(int i=0;i<shots[j].size();i++){
			ostringstream tmp;
			tmp << shots[j][i].StartTime << "\t" << shots[j][i].EndTime << "\t" << endl ;

			fprintf(output[j][0],tmp.str().c_str());
		}
		for(int i=0;i<BestN;i++){
			all_result_time_sort.push_back(shots[j][i]);
		}

		//音量ソート
		CalcVolumeAndFlowMixPriority(shots[j],1,0);										//volume_ratio,optflow_ratioの比で優先度を定義
		std::sort(shots[j].begin(), shots[j].end(), &Shot::VolumeAndFlowMixPriority_cmp);		//実際にソートする関数
		for(int i=0;i<shots[j].size();i++){
			ostringstream tmp;
			tmp << shots[j][i].StartTime << "\t" << shots[j][i].EndTime << "\t" << endl ;

			fprintf(output[j][1],tmp.str().c_str());
		}
		for(int i=0;i<BestN;i++){
			all_result_time_sort.push_back(shots[j][i]);
		}

		//スレッショルドあり音量ソート
		std::sort(shots[j].begin(), shots[j].end(), &Shot::VolumeChangeThreSum_cmp);	
		for(int i=0;i<shots[j].size();i++){
			ostringstream tmp;
			tmp << shots[j][i].StartTime << "\t" << shots[j][i].EndTime << "\t" << endl ;

			fprintf(output[j][2],tmp.str().c_str());
		}
		for(int i=0;i<BestN;i++){
			all_result_time_sort.push_back(shots[j][i]);
		}
	}
	//スレッショルドあり音量ソート
	std::sort(all_result_time_sort.begin(), all_result_time_sort.end(), &Shot::StartTime_cmp);

	double before = -1;
	for(int i=0;i<all_result_time_sort.size();i++){
		if(before != all_result_time_sort[i].StartTime){
			ostringstream tmp1,tmp2;
			tmp1 << all_result_time_sort[i].StartTime << "\t" << all_result_time_sort[i].EndTime << "\t" << endl ;
			fprintf(output_all,tmp1.str().c_str());

			tmp2 << all_result_time_sort[i].StartTime << "\t" << all_result_time_sort[i].EndTime << "\t"  << all_result_time_sort[i].FlowAverage << "\t" << all_result_time_sort[i].VolumeChangeSum << "\t" << all_result_time_sort[i].VolumeChangeThreSum <<  endl ;
			fprintf(output_alldata,tmp2.str().c_str());

		}
		before = all_result_time_sort[i].StartTime;
	}

}

*/

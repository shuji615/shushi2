#include <fstream>
#include <iostream>
#include <string>
#include <cstring>

#include <iterator>
#include <vector>
#include <algorithm>

#define MAveWidth 100
#define FPS 10.0

class Shot{
public:
	double StartTime;
	double EndTime;
	double FlowMinTime;
	
	// ”äŠrŠÖ”
    static bool cmp(Shot a, Shot b)
    {
		return (a.FlowMinTime < b.FlowMinTime); 
    } 
};

int CutandDefinePriority(char* filename){

	std::ifstream ifs(filename);
    if (ifs.fail())
    {
        std::cerr << "¸”s" << std::endl;
        return -1;
    }
 
	std::vector<double> RawData;
	std::vector<double> MAveData;
	std::copy(std::istream_iterator<double>(ifs), std::istream_iterator<double>(), std::back_inserter(RawData));

	//MAveWidth •‚Å•½‹Ï‰»‚µCMAveData‚ÉŠi”[
	//“¯‚ÉMAveData‚Ì•½‹Ï‚ÌŒvZ
	double average=0;
	for(int i=0;i<RawData.size()-MAveWidth;i++){
		for(int j=0;j<MAveWidth;j++){
			MAveData[i] += RawData[i+j];
		}
		MAveData[i] /= MAveWidth;
		average += MAveData[i];
	}
	average /= MAveData.size();
	const double AVE = 0.7 * average;

	std::vector<Shot> Shots;
	Shot tmp;

	bool flag = false;
	double ShotMin = AVE;
	for(int i=0;i<MAveData.size();i++){
		if(flag == true){
			if(MAveData[i] < ShotMin){
				ShotMin = MAveData[i];
			}
			if(MAveData[i] > AVE){
				tmp.EndTime = (double)i/FPS;
				tmp.FlowMinTime = ShotMin;
				Shots.push_back(tmp);

				flag = false;
			}
		}else if(MAveData[i] < AVE){
			tmp.StartTime = (double)i/FPS;
			flag = true;
		}
	}
	std::sort(Shots.begin(), Shots.end(), &Shot::cmp);
	std::cout << Shots[0].FlowMinTime << std::endl << Shots[1].FlowMinTime << std::endl << Shots[2].FlowMinTime << std::endl;

    return 0;
}
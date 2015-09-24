#pragma once

//class ColorWheel
//{
//public:
//	ColorWheel(void);
//	~ColorWheel(void);
//
//	int **colorwheel;
//	int ncols;
//
//private:
//	double RY, YG, GC, CB, BM, MR;
//	
//public:
//	void MakeColorWheel(void);
//};


#define M_PI 3.14159265

class ColorWheel
{
public:
	ColorWheel();
	virtual ~ColorWheel();
	void setcols(int r, int g, int b, int k);
	void makecolorwheel();
	void computeColor(const float fx, const float fy, unsigned char *pix);
	void computeColor(const double fx, const double fy, unsigned char *pix);
	int ncols ;
	int **colorwheel;
};
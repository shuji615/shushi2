#include "ColorWheel.h"
#include <math.h>



#define MAXCOLS 60
ColorWheel::ColorWheel()
{
	ncols = 0;
	colorwheel = new int*[MAXCOLS];
	for (int i=0;i<MAXCOLS;i++)
		colorwheel[i] = new int[3];
}
ColorWheel::~ColorWheel()
{
	for (int i=0;i<MAXCOLS;i++)
		delete[] colorwheel[i];
	delete[] colorwheel;
}
void ColorWheel::setcols(int r, int g, int b, int k)
{
	colorwheel[k][0] = r;
	colorwheel[k][1] = g;
	colorwheel[k][2] = b;
}

void ColorWheel::makecolorwheel()
{
	// relative lengths of color transitions:
	// these are chosen based on perceptual similarity
	// (e.g. one can distinguish more shades between red and yellow 
	//  than between yellow and green)
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;
	ncols = RY + YG + GC + CB + BM + MR;
	//printf("ncols = %d\n", ncols);
	if (ncols > MAXCOLS)
		return;
	int i;
	int k = 0;
	for (i = 0; i < RY; i++) setcols(255,	   255*i/RY,	 0,	       k++);
	for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,		 0,	       k++);
	for (i = 0; i < GC; i++) setcols(0,		   255,		 255*i/GC,     k++);
	for (i = 0; i < CB; i++) setcols(0,		   255-255*i/CB, 255,	       k++);
	for (i = 0; i < BM; i++) setcols(255*i/BM,	   0,		 255,	       k++);
	for (i = 0; i < MR; i++) setcols(255,	   0,		 255-255*i/MR, k++);
}

void ColorWheel::computeColor(const float fx, const float fy, unsigned char *pix)
{

	//if (ncols == 0)
	//	makecolorwheel();

	float rad = sqrt(fx * fx + fy * fy);
	float a = atan2(-fy, -fx) / M_PI;
	float fk = (a + 1.0) / 2.0 * (ncols-1);
	int k0 = (int)fk;
	int k1 = (k0 + 1) % ncols;
	float f = fk - k0;
	//f = 0; // uncomment to see original color wheel
	for (int b = 0; b < 3; b++) {
		float col0 = colorwheel[k0][b] / 255.0;
		float col1 = colorwheel[k1][b] / 255.0;
		float col = (1 - f) * col0 + f * col1;
		if (rad <= 1)
			col = 1 - rad * (1 - col); // increase saturation with radius
		else
			col *= .75; // out of range
		pix[2 - b] = (int)(255.0 * col);
	}
}

void ColorWheel::computeColor(const double fx, const double fy, unsigned char *pix)
{

	//if (ncols == 0)
	//	makecolorwheel();

	double rad = sqrt(fx * fx + fy * fy);
	double a = atan2(-fy, -fx) / M_PI;
	double fk = (a + 1.0) / 2.0 * (ncols-1);
	int k0 = (int)fk;
	int k1 = (k0 + 1) % ncols;
	double f = fk - k0;
	//f = 0; // uncomment to see original color wheel
	for (int b = 0; b < 3; b++) {
		float col0 = colorwheel[k0][b] / 255.0;
		float col1 = colorwheel[k1][b] / 255.0;
		float col = (1 - f) * col0 + f * col1;
		if (rad <= 1)
			col = 1 - rad * (1 - col); // increase saturation with radius
		else
			col *= .75; // out of range
		pix[2 - b] = (int)(255.0 * col);
	}
}






//ColorWheel::ColorWheel(void)
//{
//	RY = 15.0;
//	YG = 6.0;
//	GC = 4.0;
//	CB = 11.0;
//	BM = 13.0;
//	MR = 6.0;
//	
//	ncols = (int)(RY + YG + GC + CB + BM + MR);
//	colorwheel = 0;
//}
//
//ColorWheel::~ColorWheel(void)
//{
//	if(colorwheel!=0){
//		for(int i=0; i<ncols; i++){
//			delete [] colorwheel[i];
//		}
//
//		delete [] colorwheel;
//	}
//}
//
//void ColorWheel::MakeColorWheel(void)
//{
//	
//	// initialize	
//	colorwheel = new int*[ncols];
//
//	for(int i=0; i<ncols; i++){
//		colorwheel[i] = new int [3];	// r g b
//		for(int j=0; j<3; j++)
//			colorwheel[i][j] = 0;
//	}
//
//
//	int col = 0;
//	// RY
//	for(int i=0; i<RY; i++){
//		colorwheel[i][0] = (int)255.0;
//		colorwheel[i][1] = (int)floor((255.0*(double)(i)/RY));
//	}
//	col = col+(int)RY;
//
//	// YG
//	for(int i=col, j=0; i<col+YG; i++, j++){
//		colorwheel[i][0] = (int)(255.0 - floor(255.0*(double)(j)/YG));
//		colorwheel[i][1] = (int)255.0;
//	}
//	col = col+(int)YG;
//
//	// GC
//	for(int i=col, j=0; i<col+GC; i++, j++){
//		colorwheel[i][1] = (int)255.0;
//		colorwheel[i][2] = (int)(floor(255.0*(double)(j)/GC));
//	}
//	col = col+(int)GC;
//
//	// CB
//	for(int i=col, j=0; i<col+CB; i++, j++){
//		colorwheel[i][1] = (int)(255.0 - floor(255.0*(double)(j)/CB));
//		colorwheel[i][2] = (int)255.0;
//	}
//	col = col+(int)CB;
//
//	// BM
//	for(int i=col, j=0; i<col+BM; i++, j++){
//		colorwheel[i][2] = (int)255.0;
//		colorwheel[i][0] = (int)(floor(255.0*(double)(j)/BM));
//	}
//	col = col+(int)BM;
//
//	// MR
//	for(int i=col, j=0; i<col+MR; i++, j++){	
//		colorwheel[i][2] = (int)(255.0 - floor(255.0*(double)(j)/MR));
//		colorwheel[i][0] = (int)255.0;
//	}
//
//}

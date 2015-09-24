#pragma once

#include "project.h"
#include "Image.h"
#include "OpticalFlow.h"
#include <iostream>
#include "ColorWheel.h"

class OpticalFlowIO
{
	// get the parameters
	double alpha;		//1
	double ratio;			//0.5
	int minWidth;			//40
	int nOuterFPIterations; //3;
	int nInnerFPIterations;
	int nSORIterations;		//20;	
	int nCGIterations;

public:
	OpticalFlowIO(void);
	~OpticalFlowIO(void);
	void VisualizeColorMap(void);
	void VisualizeFlow(DImage & vx, DImage & vy, DImage & flow);
	void OpticalFlowIOMain(DImage & vx, DImage & vy, DImage & Im1, DImage & Im2, DImage &warpI2);
};

#include "OpticalFlowIO.h"


OpticalFlowIO::OpticalFlowIO(void)
{
	// get the parameters
	alpha= 0.012;		//1
	ratio=0.75;			//0.5
	minWidth= 20;			//40
	nOuterFPIterations = 7; //3;
	nInnerFPIterations = 1;
	nSORIterations=	30;		//20;	
	nCGIterations=40;
}



OpticalFlowIO::~OpticalFlowIO(void)
{
}




void OpticalFlowIO::OpticalFlowIOMain(DImage & vx, DImage & vy, DImage & Im1, DImage & Im2, DImage &warpI2)
{
	// DImage warpI2;
	OpticalFlow::Coarse2FineFlow(vx,vy,warpI2,Im1,Im2,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations);
	
	//このコメントアウトは自分で入れた（ビジュアライズがいらないから）
	/*
	DImage flow;
	VisualizeFlow(vx, vy, flow);
	flow.imwrite("flowcolor.jpg");
	*/
}



// convert flow (vx, vy) into color image
void OpticalFlowIO::VisualizeFlow(DImage & vx, DImage & vy, DImage & flow)
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
	printf("max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n", maxrad, minu, maxu, minv, maxv);

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

					cvLine(image, cvPointFrom32f(point1), cvPointFrom32f(point2), CV_RGB(255, 0, 0), 1, CV_AA, 0);
				}
			}
		}
	}
	cvSaveImage("arrow.jpg", image);
	cvReleaseImage(&image);


	delete [] uflow;	// release
	delete [] vflow;
}









// make color reference map of flow
void OpticalFlowIO::VisualizeColorMap(void)
{
	DImage flow;

	int width, height;
	width  = 200;
	height = 200;

	flow.allocate(width, height, 3);	// color image
	flow.setValue(0);					// initialize

	ColorWheel cw;
	cw.makecolorwheel();
	
	double maxdia = 100;
	int cx,cy;
	cx = cy = maxdia;
	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){

			double rad = sqrt((double)(y-cy)*(y-cy) + (double)(x-cx)*(x-cx));
			if(rad > maxdia){
				// do nothing
			}
			else{

				double fx = (double)(x-cx)/(double)maxdia;
				double fy = (double)(y-cy)/(double)maxdia;
				unsigned char pix[3];
				cw.computeColor(fx, fy, pix);

				for(int k=0; k<3; k++)
					flow[(y*width+x)*3+k] = pix[k];
			}
		}
	}

	flow.imwrite("flowRef.jpg");
}






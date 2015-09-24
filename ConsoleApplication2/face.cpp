#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <ctype.h>
#include <stdio.h>
#include <math.h>
 
 
#define INIT_TIME       350
#define Zeta            10.0    
 
#define LED_THRESH      80
#define LED_WIDTH       20
#define GREEN_HEIGHT    80
#define LED_SPLIT       15
#define GREEN_SPLIT     50
 
 
int sub_face_main (int argc, char **argv)
{
    
    double B_PARAM = 1.0 / 150.0;
    double T_PARAM = 1.0 / 350.0;
 
    int frame_count;
    int key;
    int repeat;
 
    int x, y, x2, y2, i, j;
    int white = 0;
    int led_width = 0;
    int start, end;
    int target_count = 0;
 
    int led_line[640] = {0};
    int led_start[50] = {0};
    int led_end[50] = {0};
    int led_wide[50] = {0};
 
    int green_height = 0;
    int obj_count = 0;
    int obj_count_new = 0;
 
    int green_line[480] = {0};
    int green_start[50] = {0};
    int green_end[50] = {0};
 
 
 
    //動画ファイル取得
    CvCapture *capture = cvCreateFileCapture( "Video.avi" );
    if(capture == NULL)
    {
        printf( "動画ファイルが見つかりませんでした。\n");
        cvWaitKey(0);
        return -1;
    }
 
 
    //入力画像のサイズ取得
    IplImage *input = cvQueryFrame( capture );
    CvSize sizeofimage = cvGetSize( input );
    int height = input->height;
    int width = input->width;
 
    
    
    //物体抽出に必要なデータ領域の準備
    IplImage *kido_average = cvCreateImage (sizeofimage, IPL_DEPTH_32F, 3);
    IplImage *OUTPUT = cvCreateImage (sizeofimage, IPL_DEPTH_8U, 1);
    IplImage *kido_ampli = cvCreateImage (sizeofimage, IPL_DEPTH_32F, 3);
    IplImage *kido_low = cvCreateImage (sizeofimage, IPL_DEPTH_32F, 3);
    IplImage *kido_top = cvCreateImage (sizeofimage, IPL_DEPTH_32F, 3);
    IplImage *input_save = cvCreateImage (sizeofimage, IPL_DEPTH_32F, 3);
    IplImage *kido_mask = cvCreateImage (sizeofimage, IPL_DEPTH_8U, 1);
    
    IplImage *obj = cvCreateImage (sizeofimage, IPL_DEPTH_8U, 3);
    IplImage *output = cvCreateImage( sizeofimage, IPL_DEPTH_8U, 1 );
 
 
    
    
    // 背景の輝度平均を計算する
    cvSetZero (kido_average);
    for (frame_count = 0; frame_count < INIT_TIME; frame_count++)
    {
        input = cvQueryFrame (capture);
        cvAcc (input, kido_average);
    }
 
    cvConvertScale (kido_average, kido_average, 1.0 / INIT_TIME);
 
    
    
    // 背景の輝度振幅を計算する
    cvSetZero (kido_ampli);
    for (frame_count = 0; frame_count < INIT_TIME; frame_count++) 
    {
        input = cvQueryFrame (capture);
        cvConvert (input, input_save);
        cvSub (input_save, kido_average, input_save);
        cvPow (input_save, input_save, 2.0);
        cvConvertScale (input_save, input_save, 2.0);
        cvPow (input_save, input_save, 0.5);
        cvAcc (input_save, kido_ampli);
    }
 
    cvConvertScale (kido_ampli, kido_ampli, 1.0 / INIT_TIME);
 
 
    //背景となりうる画素の輝度振幅の最小値を求める
    cvSub (kido_average, kido_ampli, kido_low);
    cvSubS (kido_low, cvScalarAll (Zeta), kido_low);
    
    //背景となりうる画素の輝度振幅の最大値を求める
    cvAdd (kido_average, kido_ampli, kido_top);
    cvAddS (kido_top, cvScalarAll (Zeta), kido_top);
 
 
    //ウィンドウの生成
    cvNamedWindow( "Input", CV_WINDOW_AUTOSIZE );
    cvNamedWindow( "Kido_Mask", CV_WINDOW_AUTOSIZE );
    cvNamedWindow( "Kido_Diff", CV_WINDOW_AUTOSIZE );
    //cvNamedWindow( "Obj", CV_WINDOW_AUTOSIZE );
//  cvNamedWindow( "Output", CV_WINDOW_AUTOSIZE );
 
 
 
    
 
 
 
    //動画処理のループ
    while(1)
    {
        //物体領域を表示する画面のクリア
        cvSetZero( obj );
        cvSetZero( OUTPUT );
        
        //入力画像を取得
        input = cvQueryFrame( capture );
 
        if( input == NULL )
        {
            break;
        }
 
    
        //入力画像を複製
        cvConvert( input, input_save );
 
        
        //入力画像(複製)の背景と判断された部分を黒く塗りつぶす
        cvInRange (input_save, kido_low, kido_top, kido_mask);
        cvNot( kido_mask, kido_mask );
 
        
 
//////////////////////////背景更新///////////////////////////////////////////////////////////////   
//      //輝度振幅を再計算する
//      cvSub (input_save, kido_average, input_save);
//      cvPow (input_save, input_save, 2.0);
//      cvConvertScale (input_save, input_save, 2.0);
//      cvPow (input_save, input_save, 0.5);
//
//      // 背景と判断された領域の背景の輝度平均と輝度振幅を更新する
//      cvRunningAvg (input, kido_average, B_PARAM, kido_mask);
//      cvRunningAvg (input_save, kido_ampli, B_PARAM, kido_mask);
//
//      // 物体領域と判断された領域では輝度振幅のみを（背景領域よりも遅い速度で）更新する
//      cvNot (kido_mask, kido_mask);
//      cvRunningAvg (input_save, kido_ampli, T_PARAM, kido_mask);
////////////////////////背景更新////////////////////////////////////////////////////////////////
 
 
 
        ////膨張
        //for( repeat=0; repeat<2; repeat++ )
        //{
        //  cvDilate( kido_mask, kido_mask, NULL, 1);
        //}
 
        //人物の幅を描画(赤)
        for( x=0; x<width-1; x++ )
        {
            for( y=0; y<height-1; y++ )
            {
                if( kido_mask->imageData[kido_mask->widthStep*y+x] != 0 )
                {
                    white++;
                }
            }
            if( white > LED_THRESH )
            {
                led_line[x] = 1;
            }
            else
            {
                led_line[x] = 0;
            }
            white = 0;
        }
 
        i = 0;
        target_count = 0;
        for( x=0; x<640; x++ )
        {
            if( led_line[x] == 1 )
            {
                start = x;
 
                while( led_line[x] == 1 )
                {
                    x++;
                    led_width++;
                }
 
                end = x;
 
                if( led_width < LED_WIDTH )
                {
                    for( x2=start; x2<end; x2++ )
                    {
                        led_line[x2] = 0;
                    }
                }
                else
                {
                    led_start[i] = start;
                    led_end[i] = end;
                    led_wide[i] = led_width;
 
                    i++;
                    target_count++;
                }
            }
            led_width = 0;
        }
 
        //物体が縦に分割されたらくっつける
        for( i=0; i<target_count; i++ )
        {
            if( led_start[i+1]-led_end[i] < LED_SPLIT )
            {
                if( led_wide[i] < LED_SPLIT || led_wide[i+1] < LED_SPLIT )
                {
                    for( x=led_end[i]; x<led_start[i+1]; x++ )
                    {
                        led_line[x] = 1;
                    }
                }
            }
        }
 
        //改めて物体領域のx座標を取得
        i = 0;
        target_count = 0;
        for( x=0; x<640; x++ )
        {
            if( led_line[x] == 1 )
            {
                start = x;
 
                while( led_line[x] == 1 )
                {
                    x++;
                }
 
                end = x;
 
                led_start[i] = start;
                led_end[i] = end;
                led_wide[i] = led_width;
 
                cvRectangle( obj, cvPoint(led_start[i],0), cvPoint(led_end[i],height-1), CV_RGB(255,0,0), -1 );
 
                i++;
                target_count++;
            }
            led_width = 0;
        }
 
 
        
 
        //物体の高さを描画
        obj_count = 0;
        obj_count_new = 0;
        white = 0;
        for( i=0; i<target_count; i++ )
        {
            for( y=0; y<height-1; y++ )
            {
                for( x=led_start[i]; x<led_end[i]; x++ )
                {
                    if( kido_mask->imageData[kido_mask->widthStep*y+x] != 0 )
                    {
                        white++;
                    }
 
                    if( white > LED_WIDTH-1 )
                    {
                        green_line[y] = 1;
                    }
                    else
                    {
                        green_line[y] = 0;
                    }
 
                }
                white = 0;
            }
 
            j = 0;
            for( y=0; y<480; y++ )
            {
                if( green_line[y] == 1 )
                {
                    start = y;
 
                    while( green_line[y] == 1 )
                    {
                        y++;
                    }
 
                    end = y;
 
                    
                    green_start[j] = start;
                    green_end[j] = end;
 
                    j++;
                    obj_count++;
                }
                green_height = 0;
            }
 
            //物体が横に分割したらくっつける
            for( j=0; j<obj_count; j++ )
            {
                if( green_start[j+1]-green_end[j] < GREEN_SPLIT )
                {
                    for( y=green_end[j]; y<green_start[j+1]; y++ )
                    {
                        green_line[y] = 1;
                    }
                }
            }
 
            //改めて物体領域のy座標を取得
            for( y=0; y<480; y++ )
            {
                if( green_line[y] == 1 )
                {
                    start = y;
 
                    while( green_line[y] == 1 )
                    {
                        y++;
                        green_height++;
                    }
 
                    end = y;
 
                    if( green_height < GREEN_HEIGHT )
                    {
                        for( y2=start; y2<end; y2++ )
                        {
                            green_line[y2] = 0;
                        }
                    }
                    else
                    {
                        green_start[j] = start;
                        green_end[j] = end;
 
                        cvRectangle( obj, cvPoint(led_start[i],green_start[j]), cvPoint(led_end[i],green_end[j]), CV_RGB(0,255,0), -1 );
                        cvRectangle( input, cvPoint(led_start[i],green_start[j]), cvPoint(led_end[i],green_end[j]), CV_RGB(0,255,255), 2 );
 
                        j++;
                        obj_count_new++;
                    }
                }
                green_height = 0;
            }
        }
 
        //結果の表示
        cvShowImage( "Input", input );
        cvShowImage( "Kido_Mask", kido_mask );
        cvShowImage( "Kido_Diff", OUTPUT );
        cvShowImage( "Obj", obj );
        cvShowImage( "Output", output );
        
        key = cvWaitKey (1);
 
        //[r]キーで撮影
        if(key == 0x72 )
        {
            cvSaveImage( "test_pict4.bmp", input );
        }
 
 
        //[Esc]キーで終了
        if(key == 0x1b)
        {
            break;
        }
 
        //[スペース]キーで一時停止
        else if(key == 0x20)
        {
            cvWaitKey(0);
        }
 
        //[>]キーで100フレームとばす
        else if(key == 0x3e)
        {
            for(frame_count=0; frame_count<100; frame_count++)
            {
                input = cvQueryFrame( capture );
            }
        }
                
    }
 
    //ウィンドウ，画像領域の破棄
    cvDestroyAllWindows( );
    cvReleaseCapture( &capture );
    cvReleaseImage (&kido_average);
    cvReleaseImage (&kido_ampli);
    cvReleaseImage (&kido_low);
    cvReleaseImage (&kido_top);
    cvReleaseImage (&input_save);
    cvReleaseImage (&kido_mask);
    cvReleaseImage (&output);
    cvReleaseImage (&obj);
 
    return 0;
}
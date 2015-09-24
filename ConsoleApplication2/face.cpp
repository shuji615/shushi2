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
 
 
 
    //����t�@�C���擾
    CvCapture *capture = cvCreateFileCapture( "Video.avi" );
    if(capture == NULL)
    {
        printf( "����t�@�C����������܂���ł����B\n");
        cvWaitKey(0);
        return -1;
    }
 
 
    //���͉摜�̃T�C�Y�擾
    IplImage *input = cvQueryFrame( capture );
    CvSize sizeofimage = cvGetSize( input );
    int height = input->height;
    int width = input->width;
 
    
    
    //���̒��o�ɕK�v�ȃf�[�^�̈�̏���
    IplImage *kido_average = cvCreateImage (sizeofimage, IPL_DEPTH_32F, 3);
    IplImage *OUTPUT = cvCreateImage (sizeofimage, IPL_DEPTH_8U, 1);
    IplImage *kido_ampli = cvCreateImage (sizeofimage, IPL_DEPTH_32F, 3);
    IplImage *kido_low = cvCreateImage (sizeofimage, IPL_DEPTH_32F, 3);
    IplImage *kido_top = cvCreateImage (sizeofimage, IPL_DEPTH_32F, 3);
    IplImage *input_save = cvCreateImage (sizeofimage, IPL_DEPTH_32F, 3);
    IplImage *kido_mask = cvCreateImage (sizeofimage, IPL_DEPTH_8U, 1);
    
    IplImage *obj = cvCreateImage (sizeofimage, IPL_DEPTH_8U, 3);
    IplImage *output = cvCreateImage( sizeofimage, IPL_DEPTH_8U, 1 );
 
 
    
    
    // �w�i�̋P�x���ς��v�Z����
    cvSetZero (kido_average);
    for (frame_count = 0; frame_count < INIT_TIME; frame_count++)
    {
        input = cvQueryFrame (capture);
        cvAcc (input, kido_average);
    }
 
    cvConvertScale (kido_average, kido_average, 1.0 / INIT_TIME);
 
    
    
    // �w�i�̋P�x�U�����v�Z����
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
 
 
    //�w�i�ƂȂ肤���f�̋P�x�U���̍ŏ��l�����߂�
    cvSub (kido_average, kido_ampli, kido_low);
    cvSubS (kido_low, cvScalarAll (Zeta), kido_low);
    
    //�w�i�ƂȂ肤���f�̋P�x�U���̍ő�l�����߂�
    cvAdd (kido_average, kido_ampli, kido_top);
    cvAddS (kido_top, cvScalarAll (Zeta), kido_top);
 
 
    //�E�B���h�E�̐���
    cvNamedWindow( "Input", CV_WINDOW_AUTOSIZE );
    cvNamedWindow( "Kido_Mask", CV_WINDOW_AUTOSIZE );
    cvNamedWindow( "Kido_Diff", CV_WINDOW_AUTOSIZE );
    //cvNamedWindow( "Obj", CV_WINDOW_AUTOSIZE );
//  cvNamedWindow( "Output", CV_WINDOW_AUTOSIZE );
 
 
 
    
 
 
 
    //���揈���̃��[�v
    while(1)
    {
        //���̗̈��\�������ʂ̃N���A
        cvSetZero( obj );
        cvSetZero( OUTPUT );
        
        //���͉摜���擾
        input = cvQueryFrame( capture );
 
        if( input == NULL )
        {
            break;
        }
 
    
        //���͉摜�𕡐�
        cvConvert( input, input_save );
 
        
        //���͉摜(����)�̔w�i�Ɣ��f���ꂽ�����������h��Ԃ�
        cvInRange (input_save, kido_low, kido_top, kido_mask);
        cvNot( kido_mask, kido_mask );
 
        
 
//////////////////////////�w�i�X�V///////////////////////////////////////////////////////////////   
//      //�P�x�U�����Čv�Z����
//      cvSub (input_save, kido_average, input_save);
//      cvPow (input_save, input_save, 2.0);
//      cvConvertScale (input_save, input_save, 2.0);
//      cvPow (input_save, input_save, 0.5);
//
//      // �w�i�Ɣ��f���ꂽ�̈�̔w�i�̋P�x���ςƋP�x�U�����X�V����
//      cvRunningAvg (input, kido_average, B_PARAM, kido_mask);
//      cvRunningAvg (input_save, kido_ampli, B_PARAM, kido_mask);
//
//      // ���̗̈�Ɣ��f���ꂽ�̈�ł͋P�x�U���݂̂��i�w�i�̈�����x�����x�Łj�X�V����
//      cvNot (kido_mask, kido_mask);
//      cvRunningAvg (input_save, kido_ampli, T_PARAM, kido_mask);
////////////////////////�w�i�X�V////////////////////////////////////////////////////////////////
 
 
 
        ////�c��
        //for( repeat=0; repeat<2; repeat++ )
        //{
        //  cvDilate( kido_mask, kido_mask, NULL, 1);
        //}
 
        //�l���̕���`��(��)
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
 
        //���̂��c�ɕ������ꂽ�炭������
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
 
        //���߂ĕ��̗̈��x���W���擾
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
 
 
        
 
        //���̂̍�����`��
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
 
            //���̂����ɕ��������炭������
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
 
            //���߂ĕ��̗̈��y���W���擾
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
 
        //���ʂ̕\��
        cvShowImage( "Input", input );
        cvShowImage( "Kido_Mask", kido_mask );
        cvShowImage( "Kido_Diff", OUTPUT );
        cvShowImage( "Obj", obj );
        cvShowImage( "Output", output );
        
        key = cvWaitKey (1);
 
        //[r]�L�[�ŎB�e
        if(key == 0x72 )
        {
            cvSaveImage( "test_pict4.bmp", input );
        }
 
 
        //[Esc]�L�[�ŏI��
        if(key == 0x1b)
        {
            break;
        }
 
        //[�X�y�[�X]�L�[�ňꎞ��~
        else if(key == 0x20)
        {
            cvWaitKey(0);
        }
 
        //[>]�L�[��100�t���[���Ƃ΂�
        else if(key == 0x3e)
        {
            for(frame_count=0; frame_count<100; frame_count++)
            {
                input = cvQueryFrame( capture );
            }
        }
                
    }
 
    //�E�B���h�E�C�摜�̈�̔j��
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
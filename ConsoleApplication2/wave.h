//---------------------------------------------------------------------------
#ifndef     __WAVE_H
#define     __WAVE_H

//---------------------------------------------------------------------------
#include <vector>

//---------------------------------------------------------------------------
struct WAVE_FORMAT
{
    unsigned short format_id;           //�t�H�[�}�b�gID
    unsigned short num_of_channels;     //�`�����l���� monaural=1 , stereo=2
    unsigned long  samples_per_sec;     //�P�b�Ԃ̃T���v�����C�T���v�����O���[�g(Hz)
    unsigned long  bytes_per_sec;       //�P�b�Ԃ̃f�[�^�T�C�Y
    unsigned short block_size;          //�P�u���b�N�̃T�C�Y�D8bit:nomaural=1byte , 16bit:stereo=4byte
    unsigned short bits_per_sample;     //�P�T���v���̃r�b�g�� 8bit or 16bit
};
//---------------------------------------------------------------------------
class WAVE
{
private:
    //�ꉞ�R�s�[�֎~
    WAVE(const WAVE&);                  //�R�s�[�R���X�g���N�^
    WAVE& operator=(const WAVE&);       //������Z�q��`
    
    WAVE_FORMAT fmt;                    //�t�H�[�}�b�g���
    std::vector<unsigned char> data;    //���f�[�^
    
    unsigned long sampling_size;        //�T���v�����O�M���T�C�Y(bytes)
    unsigned long data_size;            //�����f�[�^�T�C�Y
    
public:
    WAVE();
    WAVE(char *file_name);
    ~WAVE();
    
    bool load_from_file(char *file_name);
    bool save_to_file(char *file_name);
    
    void get_channel(std::vector<short> &buf,unsigned short ch);
    
    void set_channel(std::vector<short> &_ch0,WAVE_FORMAT _fmt);
    void set_channel(std::vector<short> &_ch0,std::vector<short> &_ch1,WAVE_FORMAT _fmt);
    
    WAVE_FORMAT    get_format()   { return fmt; }
    unsigned long  size()         { return data_size; }
    unsigned short channels()     { return fmt.num_of_channels; }
    unsigned long  sampling_rate(){ return fmt.samples_per_sec; }
    
    double sec(){ return data_size/(double)fmt.samples_per_sec; }
};
//---------------------------------------------------------------------------
#endif

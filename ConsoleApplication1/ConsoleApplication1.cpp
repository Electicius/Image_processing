// ConsoleApplication1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <math.h>
#include<stack>
#include <iostream>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>   
#include "windows.h"
#include "math.h"
#include <algorithm>
#include <malloc.h>
#include<vector>
using namespace std;  
using namespace cv;   
#define  length  1024
#define  height 1024
unsigned char* srcData = new unsigned char[length * height];
unsigned char* dstData = new unsigned char[length * height];
#define pi 3.14159


void Linear(Mat srcImg, Mat &dstImg, double k, double b) {

	for (int i = 0; i <srcImg.rows; i++)
	{
		for (int j = 0; j <srcImg.cols; j++)
		{
			dstImg.data[i * srcImg.cols + j] = k*srcImg.data[i * srcImg.cols + j] + b;
			if (dstImg.data[i * srcImg.cols + j] < 0)dstImg.data[i * srcImg.cols + j] == 0;
			else if (dstImg.data[i * srcImg.cols + j] > 255)dstImg.data[i * srcImg.cols + j] == 255;
		}
	}
}
//线性变换


void GaussianFilter (Mat srcImg,Mat &dstImg, int ksize, int type) {
	double sigma = 0.3*((ksize - 1)/2-1) + 0.8;
	for (int i = 0; i < srcImg.rows; i++)
	{
		for (int j = 0; j < srcImg.cols; j++)
		{
			dstImg.data[i * dstImg.cols + j] = srcImg.data[i * srcImg.cols + j];
		}
	}
	double **Templet = new double*[ksize];
	for (int i = 0; i < ksize; i++) {
		Templet[i] = new double[ksize];
	}
	double x2, y2;
	double sum = 0;
	for (int i = 0; i < ksize; i++) {
		x2 = (i - ksize / 2)*(i - ksize / 2);
		for (int j = 0; j < ksize; j++) {
			y2 = (i - ksize / 2)*(i - ksize / 2);
			Templet[i][j] = exp(-(x2 + y2) / 2 * sigma*sigma);
			sum += Templet[i][j];
		}
	}
	for (int i = 0; i < ksize; i++) {
		for (int j = 0; j < ksize; j++) {
			Templet[i][j] /= sum;
		}
	}

	if (type == 1){
		for (int i = 1; i < length - 1; i++) {
			for (int j = 1; j < height - 1; j++) {
				for (int m = 0; m < ksize; m++) {
					for (int n = 0; n < ksize; n++) {
						dstImg.data[i * dstImg.cols + j] += srcImg.data[m*srcImg.cols + n] * Templet[m][n];
					}
				}
			}
		}
	}
	else if (type == 0) {
		for (int i = 0; i < ksize; i++) {
			for (int j = 0; j < ksize; j++) {
				Templet[i][j] = 1 - Templet[i][j];
			}
		}
		for (int i = 1; i < length - 1; i++) {
			for (int j = 1; j < height - 1; j++) {
				for (int m = 0; m < ksize; m++) {
					for (int n = 0; n < ksize; n++) {
						dstImg.data[i * dstImg.cols + j] += srcImg.data[m*srcImg.cols + n] * Templet[m][n];
						if ((dstImg.data[i * dstImg.cols + j]) > 255) { dstImg.data[i * dstImg.cols + j] = 255; }
						else if (dstImg.data[i * dstImg.cols + j] < 0) { dstImg.data[i * dstImg.cols + j] = 0; }
					}
				}
			}
		}
	}
	
	delete Templet;
}//高斯滤波器
//高通低通滤波：高斯模板


void Translation(Mat srcImg,Mat &dstImg,int x, int y) {
	for (int i = 0; i < srcImg.rows; i++)
	{
		for (int j = 0; j < srcImg.cols; j++)
		{
			dstImg.data[i * dstImg.cols + j] = 255;
		}
	}
	for (int i = 0; i < srcImg.rows; i++)
	{
		if (i + x > 0 && i + x < srcImg.rows) {
			for (int j = 0; j < srcImg.cols; j++){
				if (j + y > 0 && j + y < srcImg.cols) {
					dstImg.data[i * dstImg.cols + j] = srcImg.data[(i + x)* srcImg.cols + (j + y)];
				}
			}
		}
    }
}//平移
//图像平移


void ImgScaling(Mat srcImg, Mat &dstImg, double m, double n)
{
	for (int i = 0; i < m - 1; i++) {
		for (int j = 0; j < n - 1; j++)
		{
			int a = i*srcImg.cols / m;
			int b = j*srcImg.rows / n;
			double u = i*srcImg.cols / m - a;
			double v = j*srcImg.rows / n - b;
			dstImg.data[i * dstImg.cols + j] = (1 - u)*(1 - v)*srcImg.data[a* srcImg.cols + b] + (1 - u)*v*srcImg.data[a* srcImg.cols + b + 1] + (1 - v)*u*srcImg.data[(a + 1)* srcImg.cols + b] + u*v*srcImg.data[(a + 1)* srcImg.cols + b + 1];

		}
	}
}
//双线性内插缩放


Mat imgRotate(Mat matSrc, double theta, bool direction)
{
	int nRowsSrc = matSrc.rows;
	int nColsSrc = matSrc.cols;
	// 如果是顺时针旋转
	if (!direction)
		theta = 2 * pi - theta;
	// 逆时针旋转矩阵
	float matRotate[3][3]{
		{ cos(theta), -sin(theta), 0 },
		{ sin(theta), cos(theta), 0 },
		{ 0, 0, 1 }
	};
	float pt[3][2]{
		{ 0, nRowsSrc },
		{ nColsSrc, nRowsSrc },
		{ nColsSrc, 0 }
	};
	for (int i = 0; i < 3; i++)
	{
		float x = pt[i][0] * matRotate[0][0] + pt[i][1] * matRotate[1][0];
		float y = pt[i][0] * matRotate[0][1] + pt[i][1] * matRotate[1][1];
		pt[i][0] = x;
		pt[i][1] = y;
	}
	// 计算出旋转后图像的长宽
	float fMin_x = min(min(min(pt[0][0], pt[1][0]), pt[2][0]), (float)0.0);
	float fMin_y = min(min(min(pt[0][1], pt[1][1]), pt[2][1]), (float)0.0);
	float fMax_x = max(max(max(pt[0][0], pt[1][0]), pt[2][0]), (float)0.0);
	float fMax_y = max(max(max(pt[0][1], pt[1][1]), pt[2][1]), (float)0.0);
	int nRows = cvRound(fMax_y - fMin_y + 0.5) + 1;
	int nCols = cvRound(fMax_x - fMin_x + 0.5) + 1;
	int nMin_x = cvRound(fMin_x + 0.5);
	int nMin_y = cvRound(fMin_y + 0.5);
	// 输出图像
	Mat matRet(nRows, nCols, matSrc.type(), Scalar(0));
	for (int j = 0; j < nRows; j++)
	{
		for (int i = 0; i < nCols; i++)
		{
			int x = (i + nMin_x) * matRotate[0][0] + (j + nMin_y) * matRotate[0][1];
			int y = (i + nMin_x) * matRotate[1][0] + (j + nMin_y) * matRotate[1][1];
			if (x >= 0 && x < nColsSrc && y >= 0 && y < nRowsSrc)
			{
				matRet.at<Vec3b>(j, i) = matSrc.at<Vec3b>(y, x);
			}//将原点坐标移到图片左上角
		}
	}
	return matRet;
}
//图像旋转


void SobelFilter(Mat srcImg1, Mat &dstImg,double sigma) {
	double **Templetx = new double*[3];
	for (int i = 0; i <3; i++) {
		Templetx[i] = new double[3];
	}
	Templetx[0][0] = { 1 };
	Templetx[0][1] = { 0 };
	Templetx[0][2] = { -1 };
	Templetx[1][0] = { 2 };
	Templetx[1][1] = { 0 };
	Templetx[1][2] = { -2 };
	Templetx[2][0] = { 1 };
	Templetx[2][1] = { 0 };
	Templetx[2][2] = { -1 };
	double **Templety = new double*[3];
	for (int i = 0; i <3; i++) {
		Templety[i] = new double[3];
	}
	Templety[0][0] = { 1 };
	Templety[0][1] = { 2 };
	Templety[0][2] = { 1 };
	Templety[1][0] = { 0 };
	Templety[1][1] = { 0 };
	Templety[1][2] = { 0 };
	Templety[2][0] = { -1 };
	Templety[2][1] = { -2 };
	Templety[2][2] = { -1 };

	double Gx = 0;
	double Gy = 0;
	for (int i = 1; i < srcImg1.rows - 1; i++) {
		for (int j = 1; j <srcImg1.cols - 1; j++) {
			for (int m = 0; m < 3; m++) {
				for (int n = 0; n <3; n++) {
					Gx += srcImg1.data[(i+m-1) * srcImg1.cols + (j+n-1)] * Templetx[m][n];
					Gy += srcImg1.data[(i + m-1) * srcImg1.cols + (j + n-1)] * Templety[m][n];
					//对每个点进行卷积操作
				}
			}
			if (sqrt(Gx*Gx + Gy*Gy)>sigma) {
				dstImg.data[i * dstImg.cols + j] = srcImg1.data[i * srcImg1.cols + j];
				Gx = 0;
				Gy = 0;//如果计算结果大于等于阈值，输出原图形
			}
			else {
				dstImg.data[i * dstImg.cols + j] = 255;
				Gx = 0;
			    Gy = 0;//如果计算结果小于阈值，赋值255
			};
		}
	}
	free (Templetx);
	free (Templety);
}
//sobel算子边缘提取：高通滤波


void ImageMix(Mat srcImg1, Mat srcImg2, Mat &dstImg,double sigma) {
	Mat TempImg(srcImg1.rows, srcImg1.cols, CV_8UC1,1);
	SobelFilter(srcImg1,TempImg,sigma);//高通滤波
	for (int i = 0; i < srcImg2.rows; i++)
	{
		for (int j = 0; j < srcImg2.cols; j++)
		{
			for (int k = 0; k < 3; k++) {
				if(TempImg.data[i * TempImg.cols + j]==255){ //判断是否属于边缘细节部分
					dstImg.data[i * dstImg.cols * 3 + j * 3 + k] = srcImg2.data[i * srcImg1.cols * 3 + j * 3 + k];
				}
				else dstImg.data[i * dstImg.cols * 3 + j * 3 + k] =0.9* TempImg.data[i * TempImg.cols + j]
					+0.1*srcImg2.data[i * srcImg1.cols * 3 + j * 3 + k];//细节:彩色=9:1
				if ((dstImg.data[i * dstImg.cols * 3 + j * 3 + k] > 255)) { 
					dstImg.data[i * dstImg.cols * 3 + j * 3 + k] = 255; 
				}
				else if (dstImg.data[i * dstImg.cols * 3 + j * 3 + k] < 0) {
					dstImg.data[i * dstImg.cols * 3 + j * 3 + k] = 0; 
				}//防止融合后越界

			}
		}

	}



}
//基于高通滤波的影像融合



void ImageBinaryzation(Mat srcImg,Mat &dstImg) {
	//二值化状态法
	int nThreshold = 0;
	int nNewThreshold = 0;//先定义峰谷的阈值

	int hist[256];//定义并初始化灰度统计数组hist
	memset(hist, 0, sizeof(hist));
	int IS1, IS2;//状态法仍将灰度分为两类，IS1 IS2分别表示类1类2的像素总数
	long double IP1, IP2;//分别定义类1类2的质量矩 
	double meanvalue1, meanvalue2;//分别定义 类1 类2的灰度均值；
	int IterationTimes;//定义迭代次数
	int max = 0, min = 256;//初始化灰度最大值最小值

						   //统计各个灰度的像素个数
	for (int i = 0; i < srcImg.rows; i++)
	{
		for (int j = 0; j < srcImg.cols; j++)
		{
			int grey = srcImg.data[i*srcImg.cols+j];
			if (grey > max)max = grey;
			if (grey < min)min = grey;
			hist[grey] = hist[grey] + 1;//每搜索到一个像素点的灰度值 相应灰度级数组的数值就加一
		}
	}
	nNewThreshold =(int)( (max + min) / 2);//初始化阈值


	for (int k = 0; k < 256; k++)
	{
		int value = hist[k];
		printf("灰度值%d的个数为：%d\n", k, value);
	}
	for (IterationTimes = 0; nThreshold != nNewThreshold&&IterationTimes < 100; IterationTimes++)
	{
		nThreshold = nNewThreshold;
		IP1 = 0.0;
		IP2 = 0.0;
		IS1 = 0;
		IS2 = 0;//将所有的值初始化为0

		for (int k = min; k < nThreshold; k++)//计算类1的质量矩和像素个数
		{
			IS1 = IS1 + hist[k];//循环统计像素个数
			IP1 += (long double)k*hist[k];//统计类1的质量矩
		}
		meanvalue1 = IP1 / IS1;//做加权平均  灰度值


		for (int l = nThreshold + 1; l <= max; l++)//计算类2的质量矩和像素个数
		{
			IS2 = IS2 + hist[l];//循环统计像素个数
			IP2 += (long double)l*hist[l];//统计类2的质量矩
		}
		meanvalue2 = IP2 / IS2;//计算类2均值；0

		nNewThreshold =int (meanvalue1 + meanvalue2) / 2;
	}

	for (int i = 0; i <srcImg.rows; i++)
	{
		for (int j = 0; j <srcImg.cols; j++)
		{
			if (srcImg.data[i*srcImg.cols + j] < nThreshold&&srcImg.data[i*srcImg.cols + j] > 0)dstImg.data[i*dstImg.cols + j] = 0;
			if (srcImg.data[i*srcImg.cols + j] >= nThreshold&&srcImg.data[i*srcImg.cols + j] > 0) dstImg.data[i*dstImg.cols + j] = 255;//二值化
		}
	}
}
//二值化状态法



bool ImageFeature(Mat M12) {
	struct point
	{
		int x;
		int y;
	};

	typedef struct
	{
		int index;//序号
		point lefttop;
		point rightbottom;
		int area;//面积
		int arealength;//周长
		float rectangularfit;//矩形度
		int barea;
		double roundfit;//圆形度
		int shapeindex;//形状指数
		int barealength;
		int lengthtowide;//长宽比
	}ShapeFeature;//形状特征

	int hist[256];//定义并初始化灰度统计数组hist
	memset(hist, 0, sizeof(hist));
	int nThreshold = 0;
	int nNewThreshold = 0;//定义峰谷的阈值
	int IS1, IS2;//状态法仍将灰度分为两类，IS1 IS2分别表示类1类2的像素总数
	double IP1, IP2;//分别定义类1类2的质量矩 
	double meanvalue1, meanvalue2;//分别定义 类1 类2的灰度均值；
	int IterationTimes;//定义迭代次数
	int max = 0, min = 255;//初始化灰度最大值最小值
	for (int counter8 = 0; counter8 < M12.rows; counter8++)
	{
		for (int counter9 = 0; counter9 < M12.cols; counter9++)
		{
			int grey = M12.at<uchar>(counter8, counter9);
			if (grey > max)max = grey;
			if (grey < min)min = grey;
			hist[grey] = hist[grey] + 1;//每搜索到一个像素点的灰度值 相应灰度级数组的数值就加一
		}
	}
	nNewThreshold = (int)((max + min) / 2);//给阈值初始化
	for (IterationTimes = 0; nThreshold != nNewThreshold&&IterationTimes < 100; IterationTimes++)//进行迭代不断更新阈值
	{
		nThreshold = nNewThreshold;
		IP1 = 0.0;
		IP2 = 0.0;
		IS1 = 0;
		IS2 = 0;//将所有的值初始化为0

		for (int k = min; k < nThreshold; k++)//计算类1的质量矩和像素个数
		{
			IS1 = IS1 + hist[k];//循环统计像素个数
			IP1 += (double)k*hist[k];//统计类1的质量矩
		}
		meanvalue1 = IP1 / IS1;//做加权平均  灰度均值


		for (int l = nThreshold + 1; l <= max; l++)//计算类2的质量矩和像素个数
		{
			IS2 = IS2 + hist[l];//循环统计像素个数
			IP2 += (double)l*hist[l];//统计类2的质量矩
		}
		meanvalue2 = IP2 / IS2;//计算类2均值；

		nNewThreshold = (int)((meanvalue1 + meanvalue2) / 2);//求峰谷的阈值
	}

	for (int count4 = 0; count4 < M12.rows; count4++)
	{
		for (int count5 = 0; count5 < M12.cols; count5++)
		{
			if (M12.at<uchar>(count4, count5) < nThreshold&&M12.at<uchar>(count4, count5) > 0) M12.at<uchar>(count4, count5) = 0;
			if (M12.at<uchar>(count4, count5) >= nThreshold&&M12.at<uchar>(count4, count5) > 0) M12.at<uchar>(count4, count5) = 255;
		}
	}
	imshow("二值化", M12);
	waitKey(0);

	int num;
	Mat srcImg = M12.clone();
	ShapeFeature *m_shapefeature = NULL;
	int nHeight = srcImg.rows, nWidth = srcImg.cols;
	// 分配内存
	LPBYTE m_lpImageCopy = (unsigned char *)malloc(nHeight * srcImg.step[1]);
	if (m_lpImageCopy == NULL)
	{
		SetCursor(LoadCursor(NULL, IDC_ARROW));
		printf("Memory Allocate error");
		return FALSE;
	}
	//若内存分配成功，将位图数据拷贝到新申请的内存中
	memcpy(m_lpImageCopy, srcImg.data, nHeight * srcImg.step[1] * sizeof(unsigned char));
	BOOL tt = srcImg.isContinuous();
	int depth = 1;//图像位数：8位，1字节
	int lRowBytes = nWidth * ((int)depth);
	//获取图像数据指针
	LPBYTE lpData = srcImg.data;
	LPBYTE lpOldBits = m_lpImageCopy;


	int stop = 0;
	int counter = 0;
	int present;
	int i, j, m, n, t;
	BYTE *p_temp;
	p_temp = new BYTE[lRowBytes*nHeight];//临时储存区域，大小为整张图片的字节数
	memset(p_temp, 255, lRowBytes*nHeight);//全置为白色

	const int T = 50;
	if (depth == 1)
	{
		for (i = 0; i < nWidth; i++)   * (lpData + (nHeight - 1) * lRowBytes + i) = 255;
		for (j = 0; j < nHeight; j++)   * (lpData + (nHeight - 1 - j) * lRowBytes) = 255;
		for (j = 1; j < nHeight - 1; j++)
		{
			if (stop == 1)
				break;
			for (i = 1; i < nWidth - 1; i++)
			{
				if (counter > 255)
				{
					printf("连通区域数目太多，请减少样本个数");
					stop = 1;
					return FALSE;
				}
				if (*(lpData + (nHeight - j - 1) * lRowBytes + i) < T)
				{
					if (*(lpData + (nHeight - j - 1 + 1) * lRowBytes + i + 1) < T)
					{
						*(p_temp + (nHeight - j - 1) * lRowBytes + i) = *(p_temp + (nHeight - j - 1 + 1) * lRowBytes + i + 1);
						present = *(p_temp + (nHeight - j - 1 + 1) * lRowBytes + i + 1);
						if (*(lpData + (nHeight - j - 1) * lRowBytes + i - 1) < T&& *(p_temp + (nHeight - j - 1)*lRowBytes + i - 1) != present)
						{
							int temp = *(p_temp + (nHeight - j - 1) * lRowBytes + i - 1);
							if (present > temp)
							{
								present = temp;
								temp = *(p_temp + (nHeight - j - 1 + 1) * lRowBytes + i + 1);
							}
							counter--;
							for (m = 1; m < nHeight; m++)
								for (n = 1; n < nWidth; n++)
								{
									if (*(p_temp + (nHeight - m - 1) * lRowBytes + n) == temp)
									{
										*(p_temp + (nHeight - m - 1) * lRowBytes + n) = present;
									}
									else if (*(p_temp + (nHeight - m - 1) * lRowBytes + n) > temp)
									{
										*(p_temp + (nHeight - m - 1) * lRowBytes + n) -= 1;
									}
								}
						}//end左前

						if (*(lpData + (nHeight - j - 1 + 1)*lRowBytes + i - 1) < T&&*(p_temp + (nHeight - j - 1 + 1)*lRowBytes + i - 1) != present)
						{
							counter--;
							int temp = *(p_temp + (nHeight - j - 1 + 1)*lRowBytes + i - 1);
							if (present < temp)
							{
								present = temp;
								temp = *(p_temp + (nHeight - j - 1 + 1)*lRowBytes + i - 1);
							}
							for (m = 1; m < nHeight; m++)
								for (n = 1; n < nWidth; n++)
								{
									if (*(p_temp + (nHeight - m - 1) *lRowBytes + n) == present)
									{
										*(p_temp + (nHeight - m - 1)*lRowBytes + n) = temp;
									}
									else if (*(p_temp + (nHeight - m - 1)*lRowBytes + n) > present)
									{
										*(p_temp + (nHeight - m - 1)*lRowBytes + n) -= 1;
									}
								}present = temp;
						}//end 左上
					}
					else if (*(lpData + (nHeight - j - 1 + 1)*lRowBytes + i) < T)
					{
						*(p_temp + (nHeight - j - 1)*lRowBytes + i) = *(p_temp + (nHeight - j - 1 + 1)*lRowBytes + i);
						present = *(p_temp + (nHeight - j - 1 + 1)*lRowBytes + i);
					}
					else if (*(lpData + (nHeight - j - 1 + 1)*lRowBytes + i - 1) < T)
					{
						*(p_temp + (nHeight - j - 1)*lRowBytes + i) = *(p_temp + (nHeight - j - 1 + 1)*lRowBytes + i - 1);
						present = *(p_temp + (nHeight - j - 1 + 1)*lRowBytes + i - 1);
					}
					else if (*(lpData + (nHeight - j - 1)*lRowBytes + i - 1) < T)
					{
						*(p_temp + (nHeight - j - 1)*lRowBytes + i) = *(p_temp + (nHeight - j - 1)*lRowBytes + i - 1);
						present = *(p_temp + (nHeight - j - 1)*lRowBytes + i - 1);
					}
					else
					{
						++counter;
						present = counter;
						*(p_temp + (nHeight - 1 - j)*lRowBytes + i) = present;
					}

				}
			}
		}
	}
	num = counter;
	if (m_shapefeature != NULL)
		delete[]m_shapefeature;

	m_shapefeature = new ShapeFeature[num];
	for (i = 0; i < num; i++)
	{
		m_shapefeature[i].index = i + 1;
		m_shapefeature[i].lefttop.x = nWidth;
		m_shapefeature[i].lefttop.y = nHeight;
		m_shapefeature[i].rightbottom.x = 0;
		m_shapefeature[i].rightbottom.y = 0;
		m_shapefeature[i].area = 0;
		m_shapefeature[i].arealength = 0;
		m_shapefeature[i].rectangularfit = 0.0;
		m_shapefeature[i].barea = 0;
		m_shapefeature[i].roundfit = 0;
		m_shapefeature[i].shapeindex = 0;
	}
	for (t = 1; t < num + 1; t++)
	{
		for (j = 1; j < nHeight - 1; j++)
		{
			for (i = 1; i < nWidth - 1; i++)
			{
				if (*(p_temp + (nHeight - j - 1) * lRowBytes + i) == t)
				{
					if (m_shapefeature[t - 1].lefttop.x > i)
						m_shapefeature[t - 1].lefttop.x = i;
					if (m_shapefeature[t - 1].lefttop.y > j)
						m_shapefeature[t - 1].lefttop.y = j;
					if (m_shapefeature[t - 1].rightbottom.x < i)
						m_shapefeature[t - 1].rightbottom.x = i;
					if (m_shapefeature[t - 1].rightbottom.y < j)
						m_shapefeature[t - 1].rightbottom.y = j;
				}
			}
		}
	}
	//计算面枳
	for (t = 0; t < num; t++)
		for (j = 1; j < nHeight - 1; j++)
			for (i = 1; i < nWidth - 1; i++)
				if (*(p_temp + (nHeight - j - 1)*lRowBytes + i) == t + 1)
				{
					m_shapefeature[t].area++;
				}

	//计算周长
	for (j = 1; j < nHeight - 1; j++)
		for (i = 1; i < nWidth - 1; i++)
			if (*(lpData + j * lRowBytes + i) - *(lpData + j * lRowBytes + i + 1) == 255)
				* (lpOldBits + j * lRowBytes + i + 1) = 100;
			else if (*(lpData + j * lRowBytes + i + 1) - *(lpData + j * lRowBytes + i) == 255)
				* (lpOldBits + j * lRowBytes + i) = 100;
			else if (*(lpData + j * lRowBytes + i) - *(lpData + (j + 1) * lRowBytes + i) == 255)
				* (lpOldBits + (j + 1)* lRowBytes + i) = 100;
			else if (*(lpData + (j + 1) * lRowBytes + i) - *(lpData + j * lRowBytes + i) == 255)
				* (lpOldBits + j * lRowBytes + i) = 100;

			for (t = 0; t < num; t++)
				for (i = 1; i < nWidth - 1; i++)
					for (j = 1; j < nHeight - 1; j++)
						if (*(p_temp + j * lRowBytes + i) == t + 1)
						{
							if (*(lpOldBits + j * lRowBytes + i) == 100)
								m_shapefeature[t].arealength++;
						}

			//算其他形状特征
			for (t = 0; t < num; t++)
			{
				m_shapefeature[t].barea = abs((m_shapefeature[t].lefttop.x - m_shapefeature[t].rightbottom.x - 1)*(m_shapefeature[t].lefttop.y - m_shapefeature[t].rightbottom.y - 1));
				m_shapefeature[t].barealength = 2 * abs((m_shapefeature[t].lefttop.x - m_shapefeature[t].rightbottom.x - 1) + 2 * abs(m_shapefeature[t].lefttop.y - m_shapefeature[t].rightbottom.y - 1));
				m_shapefeature[t].shapeindex = (m_shapefeature[t].arealength*m_shapefeature[t].arealength) / (m_shapefeature[t].area * 4 * 3.14159);
				m_shapefeature[t].rectangularfit = (double)m_shapefeature[t].area / m_shapefeature[t].barea;
				m_shapefeature[t].lengthtowide = (double)abs(m_shapefeature[t].lefttop.y - m_shapefeature[t].rightbottom.y - 1);
				m_shapefeature[t].roundfit = 4 * 3.1415926*(double)m_shapefeature[t].area / (double)(m_shapefeature[t].arealength*(double)m_shapefeature[t].arealength);
			}
			for (i = 0; i < num; i++)
			{
				printf("代号%d 面积%d 周长%d  圆形度%lf  连通总数量%d\n", m_shapefeature[i].index, m_shapefeature[i].area, m_shapefeature[i].arealength, m_shapefeature[i].roundfit, num);
			}
			getchar();
			return 0;
}

//形状特征提取




void main()
{
	printf("1.raw文件转bmp\n");
	FILE *fp = NULL;
	if ((fp = fopen("std.raw","r")) == NULL) {
		printf_s("打开失败");
	}
	else {
		printf_s("读取成功\n");
		fread(srcData,1,length *height,fp);
		fclose(fp);
	}
	//打开文件
	
	Mat img(length, height, CV_8UC1, 1);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.data[i * img.cols + j] = srcData[i * img.cols + j];
		}
	}//转成Mat类
	imshow("img",img);
	imwrite("img.bmp",img);
	waitKey(0);;//转成bmp 


	printf_s("2.线性变换：请输入线性变换的k和b:\n");
	double k, b;
	scanf_s("%lf%lf", &k, &b);
	Mat LinearImg(length, height, CV_8UC1, 1);
	Linear(img, LinearImg, k, b);
	imshow("灰度线性变换", LinearImg);
	imwrite("LinearImg.bmp", LinearImg);//灰度线性变换
	waitKey(0);;

	printf_s("3.高通/低通滤波\n");
	int type,ksize;
	Mat GaussianFilterImg(length, height, CV_8UC1, 1);
	printf_s("请输入高斯滤波的种类【高通(0)低通(1)】，窗口大小:\n");
	scanf_s("%d%d",&type,&ksize);
	GaussianFilter(img, GaussianFilterImg,  ksize, type);
	imshow("高（低）通滤波",GaussianFilterImg);
	imwrite("GaussianFilterImg.bmp", GaussianFilterImg);//高低通滤波
	waitKey(0);
	
	

	double m, n;
	printf("4.缩放:请输入缩放后的分辨率m*n:\n");
	scanf_s("%lf%lf", &m, &n);
	Mat ScalingImg(m, n, CV_8UC1, 1);
	ImgScaling(img, ScalingImg, m, n);
	imshow("缩放后的图片",ScalingImg);//双线性插值缩放
	waitKey(0);

	int x, y;
	printf_s("5.平移：请选择平移的横纵值（可为负）:\n");
	scanf_s("%d%d", &x, &y);
	Mat TranslationImg(ScalingImg.rows, ScalingImg.cols, CV_8UC1, 1);
	Translation(ScalingImg, TranslationImg, x, y);
	imshow("平移结果", TranslationImg);
	imwrite("TranslationImg.bmp", TranslationImg);//平移
	waitKey(0);


	double angle;
	int typeR;
	Mat srcImg = imread("exam1_write.jpg");
	imshow("原图片", srcImg);
	waitKey(0);
	printf_s("6.旋转：请输入旋转的角度(角度度制)和方向(0为顺时针1为逆时针)：");
	scanf_s("%lf%d",&angle,&typeR);
	double rad=angle*3.14/180;
	Mat RotatingImg = imgRotate(srcImg, rad, typeR);
	imshow("旋转结果", RotatingImg);
	imwrite("TranslationImg.bmp", RotatingImg);//旋转
	waitKey(0);

	Mat srcImage1 = imread("ik_beijing_p.bmp",0);
	Mat srcImage2 = imread("ik_beijing_c.bmp");
	Mat dstImage(srcImage1.rows,srcImage1.cols, CV_8UC3);
	Mat TempImg(srcImage1.rows, srcImage1.cols, CV_8UC1, 1);
	double delta;
	printf_s("7.高通影像融合：请输入高通的阈值(50-150,100为宜)：\n");
	scanf_s("%lf", &delta);
	SobelFilter(srcImage1, TempImg, delta);
	imshow("高通结果", TempImg);
	imwrite("SobelFilterImg.bmp", TempImg);
	waitKey(0);
	ImageMix(srcImage1, srcImage2, dstImage,delta);
	imshow("融合结果", dstImage);
	imwrite("MixedImg.bmp", dstImage);
	waitKey(0);//基于高通滤波的影像融合

	printf_s("8.二值化：\n");
	waitKey(0);
	Mat srcImage = imread("ik_beijing_p.bmp", 0);
	Mat BinaryImage(srcImage.rows, srcImage.cols, CV_8UC1, 1);
	ImageBinaryzation(srcImage, BinaryImage);
	imshow("二值化", BinaryImage);
	imwrite("BinaryImage.bmp", BinaryImage);
	waitKey(0);
 
	printf_s("9.连通域数目统计：\n");
    Mat srcImagex = imread("lena.jpg", 0);
	Mat dstImagex(srcImagex.rows, srcImagex.cols, CV_8UC1, 1);
	ImageBinaryzation(srcImagex, dstImagex);
	ImageFeature(dstImagex);

}


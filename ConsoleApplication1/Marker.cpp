#include "Marker.h"

/************************************************���캯��****************************************************/
Marker::Marker(): id(-1)
{
}

/************************************************��������****************************************************/
Marker::~Marker() 
{
}

/*******************************************����ȡ��ά���ں���Ϣ��**********************************************/
int Marker::getMarkerId(cv::Mat &markerImage, int &nRotations)
{
	assert(markerImage.rows == markerImage.cols);
	assert(markerImage.type() == CV_8UC1);

	cv::Mat grey = markerImage;//�����ζ�ά��ͼ��
	//threshold image
	cv::threshold(grey, grey, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	//Markers  are divided in 7x7 regions, of which the inner 5x5 belongs to marker info
	//the external border should be entirely black
	//��ȥ����Χ��һȦ��ɫ����ȡ��5x5������
	int cellSize = markerImage.rows / 7;//ÿһ������ı߳�

	// 1.�ȼ��һȦ��С�����Ƿ��Ǻ�ɫ
	for (int y = 0; y<7; ++y)
	{
		int inc = 6;
		//for first and last row, check the whole border
		if (y == 0 || y == 6) inc = 1;
		for (int x = 0; x<7; x += inc)
		{
			int cellX = x * cellSize;
			int cellY = y * cellSize;
			cv::Mat cell = grey(cv::Rect(cellX, cellY, cellSize, cellSize));//��ȡС����ͼ������Ȧ��
			//imshow("cell",cell);
			int nZ = cv::countNonZero(cell);//������㣨��ɫ�����ص����
			//cout<<"nZ"<<nZ<<endl;
			//cout<<"(cellSize*cellSize) / 2"<<" "<<(cellSize*cellSize) / 2<<endl;
			if (nZ >(cellSize*cellSize) / 2)//��ɫ���ص����һ��
			{
				//can not be a marker because the border element is not black!
				return -1;
			}
		}
	}

	// 2.��ͼ������Ϣ�����һ�� 5x5 �� Mat ��
	cv::Mat bitMatrix = cv::Mat::zeros(5, 5, CV_8UC1);//5x5ȫ�����
	//get information(for each inner square, determine if it is  black or white)  
	for (int y = 0; y<5; ++y)
	{
		for (int x = 0; x<5; ++x)
		{
			int cellX = (x + 1)*cellSize;
			int cellY = (y + 1)*cellSize;
			cv::Mat cell = grey(cv::Rect(cellX, cellY, cellSize, cellSize));//��ȡС����ͼ���ڲ�5x5��

			int nZ = cv::countNonZero(cell);//������㣨��ɫ�����ص����
			if (nZ>(cellSize*cellSize) / 2)//��ɫ����һ��
				bitMatrix.at<uchar>(y, x) = 1;//����Ϊ1
		}
	}
	//cout<<"bitMatrix:"<<" "<<bitMatrix<<endl;

	// 3.check all possible rotations
	//����Ϊ����4�ַ��÷���
	cv::Mat rotations[4];
	//���������롿�������1�ĸ���
	int distances[4];
	rotations[0] = bitMatrix;//��⵽��5x5��ת���ɱ����
	distances[0] = hammDistMarker(rotations[0]);//�������׼��Ϣ�ĺ�������
	std::pair<int, int> minDist(distances[0], 0);//��
	for (int i = 1; i<4; ++i)
	{
		//get the hamming distance to the nearest possible word
		rotations[i] = rotate(rotations[i - 1]);//��ʱ��ת90��
		distances[i] = hammDistMarker(rotations[i]);//�ٴμ�������3������ĺ�������

		if (distances[i] < minDist.first)//������С
		{
			minDist.first = distances[i];//ȡ4������������ϵ�
			minDist.second = i;
		}
	}
	//cout<<"minDist"<<" "<<minDist.first<<" "<<minDist.second<<endl;
	cout<<"mat2id(rotations[minDist.second]):"<<" "<<mat2id(rotations[minDist.second])<<endl;
	nRotations = minDist.second;//����ת����

	//819//1100110011
	if (minDist.first == 0)//��ȫƥ��
	{
		return mat2id(rotations[minDist.second]);//���ض�ά����Ϣ��975����
	}
	return -1;
}

/*****************************************Maker��ʼ����������**********************************************/
int Marker::hammDistMarker(cv::Mat bits)
{
	//maker  1
	int ids[5][5] =
	{
		{ 1, 1, 1, 1, 1 },
		{ 1, 1, 0, 1, 1 },
		{ 1, 0, 1, 0, 1 },
		{ 1, 1, 1, 1, 1 },
		{ 1, 1, 0, 1, 1 }
	};

	//maker 2
	//int ids_1[5][5] =
	//{
	//	{ 1, 1, 1, 1, 1 },
	//	{ 1, 1, 0, 1, 1 },
	//	{ 1, 0, 1, 1, 0 },
	//	{ 1, 1, 0, 1, 1 },
	//	{ 1, 1, 1, 1, 1 }
	//};

	int dist = 0;

	for (int y = 0; y<5; ++y)
	{
		float minSum = 1e5; //hamming distance to each possible word
		for (int p = 0; p<5; ++p)
		{
			float sum = 0;
			//now, count
			for (int x = 0; x<5; ++x)
			{
				sum += bits.at<uchar>(y, x) == ids[p][x] ? 0 : 1;//һ�£���ֵ0
			}
			if (minSum>sum)
				minSum = sum;
		}
		//do the and
		dist += minSum;
	}

	return dist;
}

/*********************************************�����ά����Ϣ*************************************************/
int Marker::mat2id(const cv::Mat &bits)
{
	int val = 0;
	for (int y = 0; y<5; ++y)
	{
		val <<= 1;
		if (bits.at<uchar>(y, 1)) val |= 1;
		val <<= 1;
		if (bits.at<uchar>(y, 3)) val |= 1;
	}
	return val;
}

/********************************************CVͼ����ת����*************************************************/
cv::Mat Marker::rotate(cv::Mat in)
{
	cv::Mat out;
	in.copyTo(out);
	for (int i = 0; i<in.rows; ++i)
	{
		for (int j = 0; j<in.cols; ++j)
		{
			out.at<uchar>(i, j) = in.at<uchar>(in.cols - j - 1, i);
		}
	}
	return out;

}

/*****************************************Maker��ʼ����������**********************************************/
bool operator<(const Marker &M1, const Marker&M2)
{
	return M1.id<M2.id;
}


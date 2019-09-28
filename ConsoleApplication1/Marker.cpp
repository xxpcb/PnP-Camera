#include "Marker.h"

/************************************************构造函数****************************************************/
Marker::Marker(): id(-1)
{
}

/************************************************析构函数****************************************************/
Marker::~Marker() 
{
}

/*******************************************【读取二维码内含信息】**********************************************/
int Marker::getMarkerId(cv::Mat &markerImage, int &nRotations)
{
	assert(markerImage.rows == markerImage.cols);
	assert(markerImage.type() == CV_8UC1);

	cv::Mat grey = markerImage;//正方形二维码图像
	//threshold image
	cv::threshold(grey, grey, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	//Markers  are divided in 7x7 regions, of which the inner 5x5 belongs to marker info
	//the external border should be entirely black
	//【去掉周围的一圈黑色，提取出5x5的网格】
	int cellSize = markerImage.rows / 7;//每一个方块的边长

	// 1.先监察一圈的小方块是否都是黑色
	for (int y = 0; y<7; ++y)
	{
		int inc = 6;
		//for first and last row, check the whole border
		if (y == 0 || y == 6) inc = 1;
		for (int x = 0; x<7; x += inc)
		{
			int cellX = x * cellSize;
			int cellY = y * cellSize;
			cv::Mat cell = grey(cv::Rect(cellX, cellY, cellSize, cellSize));//获取小方块图像（最外圈）
			//imshow("cell",cell);
			int nZ = cv::countNonZero(cell);//计算非零（白色）像素点个数
			//cout<<"nZ"<<nZ<<endl;
			//cout<<"(cellSize*cellSize) / 2"<<" "<<(cellSize*cellSize) / 2<<endl;
			if (nZ >(cellSize*cellSize) / 2)//白色像素点多余一半
			{
				//can not be a marker because the border element is not black!
				return -1;
			}
		}
	}

	// 2.将图像标记信息存放在一个 5x5 的 Mat 中
	cv::Mat bitMatrix = cv::Mat::zeros(5, 5, CV_8UC1);//5x5全零矩阵
	//get information(for each inner square, determine if it is  black or white)  
	for (int y = 0; y<5; ++y)
	{
		for (int x = 0; x<5; ++x)
		{
			int cellX = (x + 1)*cellSize;
			int cellY = (y + 1)*cellSize;
			cv::Mat cell = grey(cv::Rect(cellX, cellY, cellSize, cellSize));//获取小方块图像（内部5x5）

			int nZ = cv::countNonZero(cell);//计算非零（白色）像素点个数
			if (nZ>(cellSize*cellSize) / 2)//白色多于一半
				bitMatrix.at<uchar>(y, x) = 1;//设置为1
		}
	}
	//cout<<"bitMatrix:"<<" "<<bitMatrix<<endl;

	// 3.check all possible rotations
	//【因为会有4种放置方向】
	cv::Mat rotations[4];
	//【海明距离】异或结果中1的个数
	int distances[4];
	rotations[0] = bitMatrix;//检测到的5x5并转换成编码后
	distances[0] = hammDistMarker(rotations[0]);//计算与标准信息的海明距离
	std::pair<int, int> minDist(distances[0], 0);//？
	for (int i = 1; i<4; ++i)
	{
		//get the hamming distance to the nearest possible word
		rotations[i] = rotate(rotations[i - 1]);//逆时针转90度
		distances[i] = hammDistMarker(rotations[i]);//再次计算另外3个方向的海明距离

		if (distances[i] < minDist.first)//距离最小
		{
			minDist.first = distances[i];//取4个方向中最符合的
			minDist.second = i;
		}
	}
	//cout<<"minDist"<<" "<<minDist.first<<" "<<minDist.second<<endl;
	cout<<"mat2id(rotations[minDist.second]):"<<" "<<mat2id(rotations[minDist.second])<<endl;
	nRotations = minDist.second;//【旋转数】

	//819//1100110011
	if (minDist.first == 0)//完全匹配
	{
		return mat2id(rotations[minDist.second]);//返回二维码信息（975？）
	}
	return -1;
}

/*****************************************Maker初始的坐标数据**********************************************/
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
				sum += bits.at<uchar>(y, x) == ids[p][x] ? 0 : 1;//一致，则赋值0
			}
			if (minSum>sum)
				minSum = sum;
		}
		//do the and
		dist += minSum;
	}

	return dist;
}

/*********************************************计算二维码信息*************************************************/
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

/********************************************CV图像旋转函数*************************************************/
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

/*****************************************Maker初始的坐标数据**********************************************/
bool operator<(const Marker &M1, const Marker&M2)
{
	return M1.id<M2.id;
}


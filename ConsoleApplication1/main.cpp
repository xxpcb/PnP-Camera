///////////////////////////////////////////////////////////////////////////////
// main.cpp
// ========
// Example of understanding OpenGL transform matrix(GL_MODELVIEW)
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2006-03-17
// UPDATED: 2017-06-27
// MODIFIED: xxpcb�����λ����̬3D��ʾ��2018/08/13
///////////////////////////////////////////////////////////////////////////////

//===================
//2018/08/12 ����USB����ͷANC
#include <Eigen/Core>
#include <Eigen/LU>
#include <opencv2/opencv.hpp>  
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
#include "Marker.h"

using namespace std;
//using namespace cv;

vector<cv::Point3f> m_markerCorners3d;
vector<cv::Point2f> m_markerCorners2d;
cv::Size markerSize(100, 100);

cv::Mat camMatrix;
cv::Mat distCoeff;
float m_minContourLengthAllowed = 30;

cv::Mat frame;
cv::VideoCapture capture;
float pos_x = 0, pos_y = 0, pos_z = -10;
float agl_x = 0, agl_y = 0, agl_z = 0;
//=====================

#include <GLUT/glut.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "Matrices.h"
#include "Vectors.h"

using std::stringstream;
using std::cout;
using std::endl;
using std::ends;


// GLUT CALLBACK functions �ص�����
void displayCB();
void reshapeCB(int w, int h);
void timerCB(int millisec);
void idleCB();
void keyboardCB(unsigned char key, int x, int y);
void mouseCB(int button, int stat, int x, int y);
void mouseMotionCB(int x, int y);

// CALLBACK function when exit() called ///////////////////////////////////////
void exitCB();

void initGL();
int  initGLUT(int argc, char **argv);
bool initSharedMem();
void clearSharedMem();
void initLights();
void drawString(const char *str, int x, int y, float color[4], void *font);
void showInfo();
void drawAxis(float size=2.5f);
void drawGrid(float size=20.0f, float step=2.0f);
void drawGridXY(float size = 20.0f, float step = 2.0f);
void drawCamera();
//=======
void initCap(void);
void proImg(void);
//=======
Matrix4 setFrustum(float l, float r, float b, float t, float n, float f);
Matrix4 setFrustum(float fovY, float aspectRatio, float front, float back);

// constants
const int   SCREEN_WIDTH    = 400;
const int   SCREEN_HEIGHT   = 300;
const float CAMERA_DISTANCE = 10.0f;
const int   TEXT_WIDTH      = 8;//�ַ����
const int   TEXT_HEIGHT     = 13;//�ַ��߶�
const float DEG2RAD         = 3.141593f / 180;

// global variables
void *font = GLUT_BITMAP_8_BY_13;
int screenWidth;
int screenHeight;
bool mouseLeftDown;
bool mouseRightDown;
float mouseX, mouseY;
float cameraAngleX;
float cameraAngleY;
float cameraDistance;
int drawMode = 0;
Matrix4 matrixView;
Matrix4 matrixModel;
Matrix4 matrixModelView;    // = matrixView * matrixModel
Matrix4 matrixProjection;


//=======================


//�����ܳ�
float perimeter(const std::vector<cv::Point2f> &a)
{
	float sum = 0, dx, dy;
	for (size_t i = 0; i<a.size(); ++i)
	{
		size_t i2 = (i + 1) % a.size();
		dx = a[i].x - a[i2].x;
		dy = a[i].y - a[i2].y;
		sum += sqrt(dx*dx + dy*dy);
	}
	return sum;
}

//=======================[1]Ѱ�ҷ�������������======================
void findMarkerCandidates(const std::vector<std::vector<cv::Point>>& contours, std::vector<Marker>& detectedMarkers)
{
	std::vector<cv::Point>  approxCurve;
	std::vector<Marker> possibleMarkers;
	// 1.����ÿһ������, analyze if it is a paralelepiped likely to be the marker
	for (size_t i = 0; i<contours.size(); ++i)
	{
		// ��϶��������
		//          input arrary�㼯  approxCurve��ϵĵ�  ����������Ϊ����  ���ĸ���ʾΪ�պ�
		cv::approxPolyDP(contours[i], approxCurve, double(contours[i].size())*0.05, true);
		// ��Ѱ�ı�������
		if (approxCurve.size() != 4)
			continue;
		// �ж��ǲ���͹�����
		if (!cv::isContourConvex(approxCurve))
			continue;
		//�ĸ���֮��������Сֵ�˳����
		float minDist = 1e10;
		for (int i = 0; i<4; ++i)
		{
			cv::Point vec = approxCurve[i] - approxCurve[(i + 1) % 4];
			float squaredDistance = vec.dot(vec);
			minDist = std::min(minDist, squaredDistance);//ȡ������Сֵ
		}
		// Ԥ���趨����С���ֵ
		if (minDist < m_minContourLengthAllowed)
			continue;
		//�˳���������������������м���     
		Marker m;
		for (int i = 0; i<4; ++i)
		{
			m.points.push_back(cv::Point2f(approxCurve[i].x, approxCurve[i].y));
		}
		//��ʱ��Ե�����������ӵ�һ����͵ڶ����㣬������������ڵ����������ұ�������ʱ���
		cv::Point v1 = m.points[1] - m.points[0];
		cv::Point v2 = m.points[2] - m.points[0];
		double o = (v1.x * v2.y) - (v1.y * v2.x);
		//��������Ӧ�ý����ڶ����͵��ĸ����ﵽЧ��
		if (o  < 0.0)
		{
			std::swap(m.points[1], m.points[3]);
		}
		possibleMarkers.push_back(m);
	}
	//2.ȥ���ǵ��Ϊ�ӽ�������
	//����Ϊpair������������ѡ���Ե��˳�����һ��
	std::vector< std::pair<int, int> > tooNearCandidates;
	for (size_t i = 0; i<possibleMarkers.size(); ++i)
	{
		const Marker& m1 = possibleMarkers[i];
		//����߳��ľ�ֵ
		for (size_t j = i + 1; j<possibleMarkers.size(); ++j)
		{
			const Marker& m2 = possibleMarkers[j];
			float distSquared = 0;
			for (int c = 0; c<4; ++c)
			{
				cv::Point v = m1.points[c] - m2.points[c];
				distSquared += v.dot(v);
			}
			distSquared /= 4;
			//�ı������߳�
			if (distSquared < 50)
			{
				tooNearCandidates.push_back(std::pair<int, int>(i, j));
			}
		}
	}
	//������͵ĵ�vertor��Ϊ��־λ
	//3.ȥ����Ӱ��Ҫ��� - -
	std::vector<bool> removalMask(possibleMarkers.size(), false);
	//
	for (size_t i = 0; i<tooNearCandidates.size(); ++i)
	{
		float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].points);
		float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].points);
		size_t removalIndex;
		if (p1 > p2)
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;
		//ѡ�����е�һ�����˳�
		removalMask[removalIndex] = true;
	}
	//4.���ؾ��������˳�����������������Marker
	detectedMarkers.clear();
	for (size_t i = 0; i<possibleMarkers.size(); ++i)
	{
		if (!removalMask[i])
			detectedMarkers.push_back(possibleMarkers[i]);
	}
}

//=======================[2]����Ƕ�ά����Ϣ=======================
void detectMarkers(const cv::Mat& grayscale, std::vector<Marker>& detectedMarkers)
{
	cv::Mat canonicalMarker;//�淶�������Σ��Ķ�ά��ͼ
	std::vector<Marker> goodMarkers;

	// 1.͸�ӱ任��Ѱgoodmarker
	for (size_t i = 0; i<detectedMarkers.size(); ++i)
	{
		Marker& marker = detectedMarkers[i];
		//�õ�͸�ӱ任�ı任����ͨ���ĸ�����õ�����һ�������Ǳ���ڿռ��е����꣬�ڶ����������ĸ���������ꡣ
		cv::Mat M = cv::getPerspectiveTransform(marker.points, m_markerCorners2d);   //�任��ϵ����

																					 // Transform image to get a canonical marker image
																					 // ��͸�ӱ任�ɷ���ͼ��
		cv::warpPerspective(grayscale, canonicalMarker, M, markerSize);//�����ı任 ��canonicalmarker��ͼ
		threshold(canonicalMarker, canonicalMarker, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU); //OTSU determins threshold automatically.

																							//��ʾ�任�ɹ���ͼ��
		imshow("Gray Image1", canonicalMarker);//��������䣡��
											   //cout<<"canonicalMarker"<<canonicalMarker.size()<<endl;//100x100

		int nRotations;
		// ��Ǳ���ʶ����Ҫ�������������ؼ���ļ���ά���������Ϣ����������>//
		int id = Marker::getMarkerId(canonicalMarker, nRotations);
		//�ж��Ƿ����Ԥ����ά����Ϣ
		if (id != -1)
		{
			marker.id = id;
			//sort the points so that they are always in the same order no matter the camera orientation
			std::rotate(marker.points.begin(), marker.points.begin() + 4 - nRotations, marker.points.end());
			goodMarkers.push_back(marker);
			//cout << goodMarkers.data<< endl;
		}
	}
	// 2.ϸ���ǵ�
	if (goodMarkers.size() > 0)
	{
		std::vector<cv::Point2f> preciseCorners(4 * goodMarkers.size());
		for (size_t i = 0; i<goodMarkers.size(); ++i)
		{
			Marker& marker = goodMarkers[i];
			for (int c = 0; c<4; ++c)
			{
				preciseCorners[i * 4 + c] = marker.points[c];
				//�������
				//cout << preciseCorners[i * 4 + c] << endl;
			}
		}
		//ϸ���ǵ㺯��
		cv::cornerSubPix(grayscale, preciseCorners, cvSize(5, 5), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER, 30, 0.1));
		//copy back
		//��ϸ��λ�ø��Ƹ���ǽǵ�
		for (size_t i = 0; i<goodMarkers.size(); ++i)
		{
			Marker& marker = goodMarkers[i];
			for (int c = 0; c<4; ++c)
			{
				marker.points[c] = preciseCorners[i * 4 + c];
				//���������
				//cout << marker.points[c] << endl;
			}
		}
	}
	//��ֵ������һ����������
	detectedMarkers = goodMarkers;
	//cout<<"detectedMarkers.size()"<<detectedMarkers.size()<<endl;
}

//=======================[3]���������λ�á�=========================
void estimatePosition(std::vector<Marker>& detectedMarkers)
{
	for (size_t i = 0; i<detectedMarkers.size(); ++i)
	{
		Marker& m = detectedMarkers[i];
		cv::Mat Rvec;//��ת����
		cv::Mat_<float> Tvec;//ƽ������
		cv::Mat raux, taux;
		//===========��solvePnP��============
		//m_markerCorners3d��4��������ʵ����������
		cv::solvePnP(m_markerCorners3d, m.points, camMatrix, distCoeff, raux, taux, false, CV_P3P);

		raux.convertTo(Rvec, CV_32F);    //��ת����
		taux.convertTo(Tvec, CV_32F);   //ƽ������

		cv::Mat_<float> rotMat(3, 3);
		cv::Rodrigues(Rvec, rotMat);//��ת����->��ת����
									// Copy to transformation matrix

		("S dcode");
		// Since solvePnP finds camera location, w.r.t to marker pose, to get marker pose w.r.t to the camera we invert it.
		//std::cout << " Tvec ( X<-, Y ^, Z * �� ��" << Tvec.rows << "x" << Tvec.cols << std::endl;
		//std::cout << Tvec <<endl;		//ƽ�ƾ���
		//std::cout << " Rvec ( X<-, Y ^, Z * �� ��" << Rvec.rows << "x" << Rvec.cols << std::endl;
		//std::cout << rotMat << endl;    //��ת����

		//std::cout << camMatrix << endl;      //�������
		//std::cout << distCoeff << endl;      //����ϵ��

		//��ת����ת����ŷ���ǣ�
		float theta_z = atan2(rotMat[1][0], rotMat[0][0])*57.2958;
		float theta_y = atan2(-rotMat[2][0], sqrt(rotMat[2][0] * rotMat[2][0] + rotMat[2][2] * rotMat[2][2]))*57.2958;
		float theta_x = atan2(rotMat[2][1], rotMat[2][2])*57.2958;

		//void cv::cv2eigen(const Mat &rotMat, Eigen::Matrix< float, 1, Eigen::Dynamic > &R_n);

		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> R_n;
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> T_n;
		cv2eigen(rotMat, R_n);//��ת����
		cv2eigen(Tvec, T_n);//ƽ������
		Eigen::Vector3f P_oc;

		P_oc = -R_n.inverse()*T_n;//��ת��������*ƽ�ƾ���

		//std::cout << "\n��������:\n" << P_oc << std::endl;
		std::cout << "\n��������:" << std::endl;
		std::cout << "X ��" << P_oc.x() << std::endl;
		std::cout << "Y ��" << P_oc.y() << std::endl;
		std::cout << "Z ��" << P_oc.z() << std::endl;

		pos_x = P_oc.x()/10; pos_y = P_oc.y()/10; pos_z = P_oc.z()/10+30;

		//std::cout << "��ת����" << raux << std::endl;
		//std::cout << "��������" << m.points << std::endl;
		//std::cout << "�������������" << m_markerCorners3d << std::endl;

		std::cout << "��ת��:" << std::endl;
		std::cout << "theta_x ��" << theta_x << std::endl;
		std::cout << "theta_y ��" << theta_y << std::endl;
		std::cout << "theta_z ��" << theta_z << std::endl;

	    agl_x = -theta_x; agl_y = -theta_y; agl_z = -theta_z;

	}
}

//===================[1 2 3]��Ѱ������������=========================
void Marker_Detection(cv::Mat& img, vector<Marker>& detectedMarkers)
{
	cv::Mat imgGray;//�Ҷ�ͼ
	cv::Mat imgByAdptThr;//��ֵͼ
	vector<vector<cv::Point>> contours;

	//��ͼ��תΪ�Ҷ�ͼ
	cvtColor(img, imgGray, CV_BGRA2GRAY);
	//����ֵ����
	threshold(imgGray, imgByAdptThr, 160, 255, cv::THRESH_BINARY_INV);
	//������ͱ�����
	morphologyEx(imgByAdptThr, imgByAdptThr, cv::MORPH_OPEN, cv::Mat());
	morphologyEx(imgByAdptThr, imgByAdptThr, cv::MORPH_CLOSE, cv::Mat());
	//��������⡿
	std::vector<std::vector<cv::Point>> allContours;//�߽�����
	cv::findContours(imgByAdptThr, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	contours.clear();
	for (size_t i = 0; i<allContours.size(); ++i)//����ÿһ������
	{
		int contourSize = allContours[i].size();//��ǰ�����Ĵ�С���������������
		if (contourSize > 4)
		{
			contours.push_back(allContours[i]);
		}
	}
	//�жϷ�0����⵽��������
	if (contours.size())
	{
		//=========��Ѱ�ҷ���������������===========
		findMarkerCandidates(contours, detectedMarkers);
	}
	//�жϷ�0����⵽�ж�ά�룩
	if (detectedMarkers.size())
	{
		//==========������Ƕ�ά����Ϣ��==========
		detectMarkers(imgGray, detectedMarkers);//�Ҷ�ͼ����Ѱ��Ϣ
		//==========���������꡿��PnP��===========
		estimatePosition(detectedMarkers);
	}
}

//=====================
void initCap(void)
{
	//===============
	//��ά���ĸ��ǵ㣨3D������λ��mm ��ԭ��������
	m_markerCorners3d.push_back(cv::Point3f(-67.0f, -67.0f, 0));
	m_markerCorners3d.push_back(cv::Point3f(+67.0f, -67.0f, 0));    //???���Ͻ�Ϊԭ��
	m_markerCorners3d.push_back(cv::Point3f(+67.0f, +67.0f, 0));    //  4 3
	m_markerCorners3d.push_back(cv::Point3f(-67.0f, +67.0f, 0));    //  1 2

	//��ά���ĸ��ǵ㣨2D����
	m_markerCorners2d.push_back(cv::Point2f(0, 0));//0,0
	m_markerCorners2d.push_back(cv::Point2f(markerSize.width - 1, 0));//99,0
	m_markerCorners2d.push_back(cv::Point2f(markerSize.width - 1, markerSize.height - 1));//99,99
	m_markerCorners2d.push_back(cv::Point2f(0, markerSize.height - 1));//0,99

	 //������������ϵ��1
	camMatrix = (cv::Mat_<double>(3, 3) << 901.90912416, 0, 327.71932113,
		0, 901.67922207, 200.72983818,
		0, 0, 1);
	distCoeff = (cv::Mat_<double>(5, 1) << 0.18210939, -2.71537713, -0.00780085, 0.00095477, 16.25707729);

	int color_width = 640; //color
	int color_height = 480;

	capture.open(1);
	if (!capture.isOpened())
	{
		cout << "���ܳ�ʼ������ͷ\n";
	}
}
//=====================
void proImg(void)
{
	capture >> frame;
	//ʹ�ö�ά�����
	vector<Marker> detectedM;//��ά�����
	//===============��Ѱ���������������꡿===========
	Marker_Detection(frame, detectedM);

	for (int marknum = 0; marknum < detectedM.size(); ++marknum)//��������
	{
		for (int c = 0; c < 4; ++c)//4������
		{
			//����ǵ�����
			//cout << "(x, y)    =\t" << detectedMarkers[marknum].points[c] << endl;

			//��ǽǵ㣨������ߣ�
			cv::circle(frame, detectedM[marknum].points[c], 5, cv::Scalar(0, 0, 255), -1, 2);
			cv::line(frame, detectedM[marknum].points[(c + 1) % 4], detectedM[marknum].points[c], cv::Scalar(0, 255, 0), 1, 8);
		}
	}
	cv::imshow("suoxiao", frame);
}
//==================


///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // init global vars
    initSharedMem();

    // register exit callback
    atexit(exitCB);

    // init GLUT and GL
    initGLUT(argc, argv);
    initGL();

	//===
	initCap();

    // the last GLUT call (LOOP)
    // window will be shown and display callback is triggered by events
    // NOTE: this call never return main().
    glutMainLoop(); /* Start GLUT event-processing loop */

    return 0;
}


///////////////////////////////////////////////////////////////////////////////
// draw a grid on XZ-plane ��������㡿
///////////////////////////////////////////////////////////////////////////////
void drawGrid(float size, float step)//Ĭ��size=10.0f, step=1.0f
{
    // disable lighting ʧ�ܹ���
    glDisable(GL_LIGHTING);

    // 20x20 grid
    glBegin(GL_LINES);

    glColor3f(0.5f, 0.5f, 0.5f);//������ɫ����ɫ��
    for(float i=step; i <= size; i+= step)
    {
		//ƽ����X�ᣨ�죩�����ߵ�4������
        glVertex3f(-size, 0,  i);   // lines parallel to X-axis���죩
        glVertex3f( size, 0,  i);
        glVertex3f(-size, 0, -i);   // lines parallel to X-axis
        glVertex3f( size, 0, -i);

		//ƽ����Z�ᣨ���������ߵ�4������
        glVertex3f( i, 0, -size);   // lines parallel to Z-axis������
        glVertex3f( i, 0,  size);
        glVertex3f(-i, 0, -size);   // lines parallel to Z-axis
        glVertex3f(-i, 0,  size);
    }

    // x-axis
    glColor3f(1, 0, 0);//���죩
    glVertex3f(-size, 0, 0);
    glVertex3f( size, 0, 0);

    // z-axis
    glColor3f(0,0,1);//������
    glVertex3f(0, 0, -size);
    glVertex3f(0, 0,  size);

    glEnd();

	//===dcj
	// draw arrows(actually big square dots)
	glPointSize(10);//��Ĵ�С
	glBegin(GL_POINTS);//����
	glColor3f(1, 0, 0);//��
	glVertex3f(size, 0, 0);
	glColor3f(0, 0, 1);//��
	glVertex3f(0, 0, size);
	glEnd();
	glPointSize(1);
	//===

    // enable lighting back
    glEnable(GL_LIGHTING);
}
///////////////////////////////////////////////////////////////////////////////
// draw a grid on XY-plane ���������XY��
///////////////////////////////////////////////////////////////////////////////
void drawGridXY(float size, float step)//Ĭ��size=10.0f, step=1.0f
{
	// disable lighting ʧ�ܹ���
	glDisable(GL_LIGHTING);

	// 20x20 grid
	glBegin(GL_LINES);

	glColor3f(0.5f, 0.5f, 0.5f);//������ɫ����ɫ��
	for (float i = step; i <= size; i += step)
	{
		//ƽ����X�ᣨ�죩�����ߵ�4������
		glVertex3f(-size, i, 0);   // lines parallel to X-axis���죩
		glVertex3f(size, i, 0);
		glVertex3f(-size, -i, 0);   // lines parallel to X-axis
		glVertex3f(size, -i, 0);

		//ƽ����Y�ᣨ�̣������ߵ�4������
		glVertex3f(i, -size, 0);   // lines parallel to Y-axis���̣�
		glVertex3f(i, size, 0);
		glVertex3f(-i, -size, 0);   // lines parallel to Y-axis
		glVertex3f(-i, size, 0);
	}

	// x-axis
	glColor3f(1, 0, 0);//���죩
	glVertex3f(-size, 0, 0);
	glVertex3f(size, 0, 0);

	// y-axis
	glColor3f(0, 1, 0);//���̣�
	glVertex3f(0, -size, 0);
	glVertex3f(0, size, 0);

	glEnd();

	//===dcj
	// draw arrows(actually big square dots)
	glPointSize(10);//��Ĵ�С
	glBegin(GL_POINTS);//����
	glColor3f(1, 0, 0);//��
	glVertex3f(size, 0, 0);
	glColor3f(0, 1, 0);//��
	glVertex3f(0, size, 0);
	glEnd();
	glPointSize(1);
	//===

	// enable lighting back
	glEnable(GL_LIGHTING);
}

///////////////////////////////////////////////////////////////////////////////
// draw the local axis of an object ���������ϵ��᡿
///////////////////////////////////////////////////////////////////////////////
void drawAxis(float size)//Ĭ��size=2.5f
{
    glDepthFunc(GL_ALWAYS);     // to avoid visual artifacts with grid lines
    glDisable(GL_LIGHTING);

    // draw axis
    glLineWidth(3);//�߿�
    glBegin(GL_LINES);//����
        glColor3f(1, 0, 0);//��
        glVertex3f(0, 0, 0);
        glVertex3f(size, 0, 0);
        glColor3f(0, 1, 0);//��
        glVertex3f(0, 0, 0);
        glVertex3f(0, size, 0);
        glColor3f(0, 0, 1);//��
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, size);
    glEnd();
    glLineWidth(1);

    // draw arrows(actually big square dots)
    glPointSize(5);//��Ĵ�С
    glBegin(GL_POINTS);//����
        glColor3f(1, 0, 0);//��
        glVertex3f(size, 0, 0);
        glColor3f(0, 1, 0);//��
        glVertex3f(0, size, 0);
        glColor3f(0, 0, 1);//��
        glVertex3f(0, 0, size);
    glEnd();
    glPointSize(1);

    // restore default settings
    glEnable(GL_LIGHTING);
    glDepthFunc(GL_LEQUAL);
}


///////////////////////////////////////////////////////////////////////////////
// draw a camera ��������ģ�͡�
///////////////////////////////////////////////////////////////////////////////

// vertices for camera
static GLfloat cameraVertices[] = {
	0.500000f, -0.350000f, 0.000000f, 0.500000f, -0.350000f, 0.000000f, 0.500000f, -0.350000f, 0.000000f,
	-0.500000f, -0.350000f, 0.000000f, -0.500000f, -0.350000f, 0.000000f,
	-0.500000f, -0.350000f, 0.000000f, -0.500000f, 0.350000f, 0.000000f,
	-0.500000f, 0.350000f, 0.000000f, -0.500000f, 0.350000f, 0.000000f,
	0.500000f, 0.350000f, 0.000000f, 0.500000f, 0.350000f, 0.000000f,
	0.500000f, 0.350000f, 0.000000f, -0.500000f, 0.350000f, 0.300000f,
	-0.500000f, 0.350000f, 0.300000f, -0.500000f, 0.350000f, 0.300000f,
	0.500000f, 0.350000f, 0.300000f, 0.500000f, 0.350000f, 0.300000f,
	0.500000f, 0.350000f, 0.300000f, -0.500000f, -0.350000f, 0.300000f,
	-0.500000f, -0.350000f, 0.300000f, -0.500000f, -0.350000f, 0.300000f,
	0.500000f, -0.350000f, 0.300000f, 0.500000f, -0.350000f, 0.300000f,
	0.500000f, -0.350000f, 0.300000f, -0.285317f, 0.0927050f, 0.000000f,
	-0.242705f, 0.176336f, 0.000000f, -0.242705f, 0.176336f, -0.300000f,
	-0.242705f, 0.176336f, -0.300000f, -0.285317f, 0.0927050f, -0.300000f,
	-0.285317f, 0.0927050f, -0.300000f, -0.176336f, 0.242705f, 0.000000f,
	-0.176336f, 0.242705f, -0.300000f, -0.176336f, 0.242705f, -0.300000f,
	-0.0927050f, 0.285317f, 0.000000f, -0.0927050f, 0.285317f, -0.300000f,
	-0.0927050f, 0.285317f, -0.300000f, 0.000000f, 0.300000f, 0.000000f,
	0.000000f, 0.300000f, -0.300000f, 0.000000f, 0.300000f, -0.300000f,
	0.0927050f, 0.285317f, 0.000000f, 0.0927050f, 0.285317f, -0.300000f,
	0.0927050f, 0.285317f, -0.300000f, 0.176336f, 0.242705f, 0.000000f,
	0.176336f, 0.242705f, -0.300000f, 0.176336f, 0.242705f, -0.300000f,
	0.242705f, 0.176336f, 0.000000f, 0.242705f, 0.176336f, -0.300000f,
	0.242705f, 0.176336f, -0.300000f, 0.285317f, 0.0927050f, 0.000000f,
	0.285317f, 0.0927050f, -0.300000f, 0.285317f, 0.0927050f, -0.300000f,
	0.300000f, 0.000000f, 0.000000f, 0.300000f, 0.000000f, -0.300000f,
	0.300000f, 0.000000f, -0.300000f, 0.285317f, -0.0927050f, 0.000000f,
	0.285317f, -0.0927050f, -0.300000f, 0.285317f, -0.0927050f, -0.300000f,
	0.242705f, -0.176336f, 0.000000f, 0.242705f, -0.176336f, -0.300000f,
	0.242705f, -0.176336f, -0.300000f, 0.176336f, -0.242705f, 0.000000f,
	0.176336f, -0.242705f, -0.300000f, 0.176336f, -0.242705f, -0.300000f,
	0.0927050f, -0.285317f, 0.000000f, 0.0927050f, -0.285317f, -0.300000f,
	0.0927050f, -0.285317f, -0.300000f, 0.000000f, -0.300000f, 0.000000f,
	0.000000f, -0.300000f, -0.300000f, 0.000000f, -0.300000f, -0.300000f,
	-0.0927050f, -0.285317f, 0.000000f, -0.0927050f, -0.285317f, -0.300000f,
	-0.0927050f, -0.285317f, -0.300000f, -0.176336f, -0.242705f, 0.000000f,
	-0.176336f, -0.242705f, -0.300000f, -0.176336f, -0.242705f, -0.300000f,
	-0.242705f, -0.176336f, 0.000000f, -0.242705f, -0.176336f, -0.300000f,
	-0.242705f, -0.176336f, -0.300000f, -0.285317f, -0.0927050f, 0.000000f,
	-0.285317f, -0.0927050f, -0.300000f, -0.285317f, -0.0927050f, -0.300000f,
	-0.300000f, 0.000000f, 0.000000f, -0.300000f, 0.000000f, -0.300000f,
	-0.300000f, 0.000000f, -0.300000f, -0.194164f, 0.141069f, -0.300000f,
	-0.194164f, 0.141069f, -0.300000f, -0.228254f, 0.0741640f, -0.300000f,
	-0.228254f, 0.0741640f, -0.300000f, -0.141069f, 0.194164f, -0.300000f,
	-0.141069f, 0.194164f, -0.300000f, -0.0741640f, 0.228254f, -0.300000f,
	-0.0741640f, 0.228254f, -0.300000f, 0.000000f, 0.240000f, -0.300000f,
	0.000000f, 0.240000f, -0.300000f, 0.0741640f, 0.228254f, -0.300000f,
	0.0741640f, 0.228254f, -0.300000f, 0.141069f, 0.194164f, -0.300000f,
	0.141069f, 0.194164f, -0.300000f, 0.194164f, 0.141069f, -0.300000f,
	0.194164f, 0.141069f, -0.300000f, 0.228254f, 0.0741640f, -0.300000f,
	0.228254f, 0.0741640f, -0.300000f, 0.240000f, 0.000000f, -0.300000f,
	0.240000f, 0.000000f, -0.300000f, 0.228254f, -0.0741640f, -0.300000f,
	0.228254f, -0.0741640f, -0.300000f, 0.194164f, -0.141069f, -0.300000f,
	0.194164f, -0.141069f, -0.300000f, 0.141069f, -0.194164f, -0.300000f,
	0.141069f, -0.194164f, -0.300000f, 0.0741640f, -0.228254f, -0.300000f,
	0.0741640f, -0.228254f, -0.300000f, 0.000000f, -0.240000f, -0.300000f,
	0.000000f, -0.240000f, -0.300000f, -0.0741640f, -0.228254f, -0.300000f,
	-0.0741640f, -0.228254f, -0.300000f, -0.141068f, -0.194164f, -0.300000f,
	-0.141068f, -0.194164f, -0.300000f, -0.194164f, -0.141068f, -0.300000f,
	-0.194164f, -0.141068f, -0.300000f, -0.228254f, -0.0741640f, -0.300000f,
	-0.228254f, -0.0741640f, -0.300000f, -0.240000f, 0.000000f, -0.300000f,
	-0.240000f, 0.000000f, -0.300000f, -0.228254f, 0.0741640f, 0.000000f,
	-0.194164f, 0.141069f, 0.000000f, -0.141069f, 0.194164f, 0.000000f,
	-0.0741640f, 0.228254f, 0.000000f, 0.000000f, 0.240000f, 0.000000f,
	0.0741640f, 0.228254f, 0.000000f, 0.141069f, 0.194164f, 0.000000f,
	0.194164f, 0.141069f, 0.000000f, 0.228254f, 0.0741640f, 0.000000f,
	0.240000f, 0.000000f, 0.000000f, 0.228254f, -0.0741640f, 0.000000f,
	0.194164f, -0.141069f, 0.000000f, 0.141069f, -0.194164f, 0.000000f,
	0.0741640f, -0.228254f, 0.000000f, 0.000000f, -0.240000f, 0.000000f,
	-0.0741640f, -0.228254f, 0.000000f, -0.141068f, -0.194164f, 0.000000f,
	-0.194164f, -0.141068f, 0.000000f, -0.228254f, -0.0741640f, 0.000000f,
	-0.240000f, 0.000000f, 0.000000f, 0.306365f, 0.350000f, 0.164697f,
	0.313467f, 0.350000f, 0.178636f, 0.313467f, 0.370000f, 0.178636f,
	0.313467f, 0.370000f, 0.178636f, 0.306365f, 0.370000f, 0.164697f,
	0.306365f, 0.370000f, 0.164697f, 0.324529f, 0.350000f, 0.189697f,
	0.324529f, 0.370000f, 0.189697f, 0.324529f, 0.370000f, 0.189697f,
	0.338467f, 0.350000f, 0.196799f, 0.338467f, 0.370000f, 0.196799f,
	0.338467f, 0.370000f, 0.196799f, 0.353918f, 0.350000f, 0.199246f,
	0.353918f, 0.370000f, 0.199246f, 0.353918f, 0.370000f, 0.199246f,
	0.369369f, 0.350000f, 0.196799f, 0.369369f, 0.370000f, 0.196799f,
	0.369369f, 0.370000f, 0.196799f, 0.383307f, 0.350000f, 0.189697f,
	0.383307f, 0.370000f, 0.189697f, 0.383307f, 0.370000f, 0.189697f,
	0.394369f, 0.350000f, 0.178636f, 0.394369f, 0.370000f, 0.178636f,
	0.394369f, 0.370000f, 0.178636f, 0.401471f, 0.350000f, 0.164697f,
	0.401471f, 0.370000f, 0.164697f, 0.401471f, 0.370000f, 0.164697f,
	0.403918f, 0.350000f, 0.149246f, 0.403918f, 0.370000f, 0.149246f,
	0.403918f, 0.370000f, 0.149246f, 0.401471f, 0.350000f, 0.133795f,
	0.401471f, 0.370000f, 0.133795f, 0.401471f, 0.370000f, 0.133795f,
	0.394369f, 0.350000f, 0.119857f, 0.394369f, 0.370000f, 0.119857f,
	0.394369f, 0.370000f, 0.119857f, 0.383307f, 0.350000f, 0.108795f,
	0.383307f, 0.370000f, 0.108795f, 0.383307f, 0.370000f, 0.108795f,
	0.369369f, 0.350000f, 0.101693f, 0.369369f, 0.370000f, 0.101693f,
	0.369369f, 0.370000f, 0.101693f, 0.353918f, 0.350000f, 0.0992460f,
	0.353918f, 0.370000f, 0.0992460f, 0.353918f, 0.370000f, 0.0992460f,
	0.338467f, 0.350000f, 0.101693f, 0.338467f, 0.370000f, 0.101693f,
	0.338467f, 0.370000f, 0.101693f, 0.324529f, 0.350000f, 0.108795f,
	0.324529f, 0.370000f, 0.108795f, 0.324529f, 0.370000f, 0.108795f,
	0.313467f, 0.350000f, 0.119857f, 0.313467f, 0.370000f, 0.119857f,
	0.313467f, 0.370000f, 0.119857f, 0.306365f, 0.350000f, 0.133795f,
	0.306365f, 0.370000f, 0.133795f, 0.306365f, 0.370000f, 0.133795f,
	0.303918f, 0.350000f, 0.149246f, 0.303918f, 0.370000f, 0.149246f,
	0.303918f, 0.370000f, 0.149246f, 0.353918f, 0.370000f, 0.149246f
};
static GLfloat cameraNormals[] = {
	1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, -1.00000f, 0.000000f, 0.000000f,
	0.000000f, 1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	1.00000f, 0.000000f, 0.000000f, 0.000000f, 1.00000f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, -1.00000f, 0.000000f, 0.000000f,
	0.000000f, 0.000000f, 1.00000f, 0.000000f, 1.00000f, 0.000000f,
	1.00000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.00000f,
	0.000000f, 1.00000f, 0.000000f, -1.00000f, 0.000000f, 0.000000f,
	0.000000f, -1.00000f, 0.000000f, 0.000000f, 0.000000f, 1.00000f,
	1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f, 0.000000f,
	0.000000f, 0.000000f, 1.00000f, -0.951057f, 0.309016f, 0.000000f,
	-0.809017f, 0.587785f, 0.000000f, -0.809017f, 0.587785f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, -0.951057f, 0.309016f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, -0.587785f, 0.809017f, 0.000000f,
	-0.587785f, 0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.309017f, 0.951057f, 0.000000f, -0.309017f, 0.951057f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.000000f, 1.00000f, 0.000000f,
	0.000000f, 1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.309017f, 0.951056f, 0.000000f, 0.309017f, 0.951056f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.587785f, 0.809017f, 0.000000f,
	0.587785f, 0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.809017f, 0.587785f, 0.000000f, 0.809017f, 0.587785f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.951057f, 0.309017f, 0.000000f,
	0.951057f, 0.309017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	1.00000f, 0.000000f, 0.000000f, 1.00000f, 0.000000f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.951057f, -0.309017f, 0.000000f,
	0.951057f, -0.309017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.809017f, -0.587785f, 0.000000f, 0.809017f, -0.587785f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.587785f, -0.809017f, 0.000000f,
	0.587785f, -0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.309017f, -0.951057f, 0.000000f, 0.309017f, -0.951057f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.000000f, -1.00000f, 0.000000f,
	0.000000f, -1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.309017f, -0.951056f, 0.000000f, -0.309017f, -0.951056f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, -0.587785f, -0.809017f, 0.000000f,
	-0.587785f, -0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.809017f, -0.587785f, 0.000000f, -0.809017f, -0.587785f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, -0.951057f, -0.309017f, 0.000000f,
	-0.951057f, -0.309017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-1.00000f, -1.00000e-006f, 0.000000f, -1.00000f, -1.00000e-006f, 0.000000f,
	0.000000f, 0.000000f, -1.00000f, 0.000000f, 0.000000f, -1.00000f,
	0.809017f, -0.587785f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.951057f, -0.309016f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.587785f, -0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.309017f, -0.951056f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.000000f, -1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.309017f, -0.951056f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.587785f, -0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.809017f, -0.587785f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.951057f, -0.309017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-1.00000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.951057f, 0.309017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.809017f, 0.587785f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.587785f, 0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	-0.309017f, 0.951057f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.000000f, 1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.309017f, 0.951056f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.587785f, 0.809017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.809017f, 0.587785f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.951057f, 0.309017f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	1.00000f, 1.00000e-006f, 0.000000f, 0.951057f, -0.309016f, 0.000000f,
	0.809017f, -0.587785f, 0.000000f, 0.587785f, -0.809017f, 0.000000f,
	0.309017f, -0.951056f, 0.000000f, 0.000000f, -1.00000f, 0.000000f,
	-0.309017f, -0.951056f, 0.000000f, -0.587785f, -0.809017f, 0.000000f,
	-0.809017f, -0.587785f, 0.000000f, -0.951057f, -0.309017f, 0.000000f,
	-1.00000f, 0.000000f, 0.000000f, -0.951057f, 0.309017f, 0.000000f,
	-0.809017f, 0.587785f, 0.000000f, -0.587785f, 0.809017f, 0.000000f,
	-0.309017f, 0.951057f, 0.000000f, 0.000000f, 1.00000f, 0.000000f,
	0.309017f, 0.951056f, 0.000000f, 0.587785f, 0.809017f, 0.000000f,
	0.809017f, 0.587785f, 0.000000f, 0.951057f, 0.309017f, 0.000000f,
	1.00000f, 1.00000e-006f, 0.000000f, -0.951057f, 0.000000f, 0.309017f,
	-0.809017f, 0.000000f, 0.587786f, -0.809017f, 0.000000f, 0.587786f,
	0.000000f, 1.00000f, -1.00000e-006f, -0.951057f, 0.000000f, 0.309017f,
	0.000000f, 1.00000f, -2.00000e-006f, -0.587785f, 0.000000f, 0.809017f,
	-0.587785f, 0.000000f, 0.809017f, 0.000000f, 1.00000f, 0.000000f,
	-0.309016f, 0.000000f, 0.951057f, -0.309016f, 0.000000f, 0.951057f,
	0.000000f, 1.00000f, 0.000000f, 1.00000e-006f, 0.000000f, 1.00000f,
	1.00000e-006f, 0.000000f, 1.00000f, 0.000000f, 1.00000f, 0.000000f,
	0.309018f, 0.000000f, 0.951056f, 0.309018f, 0.000000f, 0.951056f,
	0.000000f, 1.00000f, 0.000000f, 0.587785f, 0.000000f, 0.809017f,
	0.587785f, 0.000000f, 0.809017f, 0.000000f, 1.00000f, 0.000000f,
	0.809017f, 0.000000f, 0.587786f, 0.809017f, 0.000000f, 0.587786f,
	0.000000f, 1.00000f, 0.000000f, 0.951057f, 0.000000f, 0.309017f,
	0.951057f, 0.000000f, 0.309017f, 0.000000f, 1.00000f, 1.00000e-006f,
	1.00000f, 0.000000f, 0.000000f, 1.00000f, 0.000000f, 0.000000f,
	0.000000f, 1.00000f, 2.00000e-006f, 0.951057f, 0.000000f, -0.309017f,
	0.951057f, 0.000000f, -0.309017f, 0.000000f, 1.00000f, 2.00000e-006f,
	0.809017f, 0.000000f, -0.587786f, 0.809017f, 0.000000f, -0.587786f,
	0.000000f, 1.00000f, 1.00000e-006f, 0.587785f, 0.000000f, -0.809017f,
	0.587785f, 0.000000f, -0.809017f, 0.000000f, 1.00000f, 0.000000f,
	0.309017f, 0.000000f, -0.951056f, 0.309017f, 0.000000f, -0.951056f,
	0.000000f, 1.00000f, 0.000000f, 0.000000f, 0.000000f, -1.00000f,
	0.000000f, 0.000000f, -1.00000f, 0.000000f, 1.00000f, 0.000000f,
	-0.309017f, 0.000000f, -0.951056f, -0.309017f, 0.000000f, -0.951056f,
	0.000000f, 1.00000f, 0.000000f, -0.587786f, 0.000000f, -0.809017f,
	-0.587786f, 0.000000f, -0.809017f, 0.000000f, 1.00000f, 0.000000f,
	-0.809017f, 0.000000f, -0.587785f, -0.809017f, 0.000000f, -0.587785f,
	0.000000f, 1.00000f, 0.000000f, -0.951056f, 0.000000f, -0.309018f,
	-0.951056f, 0.000000f, -0.309018f, 0.000000f, 1.00000f, -1.00000e-006f,
	-1.00000f, 0.000000f, -1.00000e-006f, -1.00000f, 0.000000f, -1.00000e-006f,
	0.000000f, 1.00000f, -2.00000e-006f, 0.000000f, 1.00000f, 0.000000f
};
static GLint cameraIndices[] = {
	2, 5, 11, 5, 8, 10, 7, 17, 7, 14, 16, 13, 23, 13, 20, 22,
	19, 1, 19, 4, 3, 18, 6, 18, 12, 21, 0, 15, 0, 9, 203, 149,
	204, 147, 204, 152, 204, 155, 204, 158, 204, 161, 204, 164, 204, 167, 204, 170,
	204, 173, 204, 176, 204, 179, 204, 182, 204, 185, 204, 188, 204, 191, 204, 194,
	204, 197, 204, 200, 203, 144, 148, 144, 202, 201, 199, 198, 196, 195, 193, 192,
	190, 189, 187, 186, 184, 183, 181, 180, 178, 177, 175, 174, 172, 171, 169, 168,
	166, 165, 163, 162, 160, 159, 157, 156, 154, 153, 151, 150, 146, 145, 148, 145,
	144, 123, 87, 124, 87, 125, 85, 126, 89, 127, 91, 128, 93, 129, 95, 130,
	97, 131, 99, 132, 101, 133, 103, 134, 105, 135, 107, 136, 109, 137, 111, 138,
	113, 139, 115, 140, 117, 141, 119, 142, 121, 143, 123, 143, 124, 29, 86, 29,
	122, 83, 120, 80, 118, 77, 116, 74, 114, 71, 112, 68, 110, 65, 108, 62,
	106, 59, 104, 56, 102, 53, 100, 50, 98, 47, 96, 44, 94, 41, 92, 38,
	90, 35, 88, 32, 84, 27, 86, 27, 29, 24, 28, 24, 82, 81, 79, 78,
	76, 75, 73, 72, 70, 69, 67, 66, 64, 63, 61, 60, 58, 57, 55, 54,
	52, 51, 49, 48, 46, 45, 43, 42, 40, 39, 37, 36, 34, 33, 31, 30,
	26, 25, 28, 25, 24
};
void drawCamera()
{
	float shininess = 32.0f;
	float ambientColor[4] = { 0.3f, 0.3f, 0.3f, 1.0f };
	float diffuseColor[4] = { 0.8f, 0.8f, 0.8f, 1.0f };
	float specularColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

	// set specular and shiniess using glMaterial
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess); // range 0 ~ 128
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specularColor);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuseColor);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambientColor);

	// set ambient and diffuse color using glColorMaterial (gold-yellow)
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glColor3fv(diffuseColor);

	// start to render polygons
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);

	glNormalPointer(GL_FLOAT, 0, cameraNormals);
	glVertexPointer(3, GL_FLOAT, 0, cameraVertices);

	glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_INT, &cameraIndices[0]);
	glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_INT, &cameraIndices[5]);
	glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_INT, &cameraIndices[10]);
	glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_INT, &cameraIndices[15]);
	glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_INT, &cameraIndices[20]);
	glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_INT, &cameraIndices[25]);
	glDrawElements(GL_TRIANGLE_STRIP, 39, GL_UNSIGNED_INT, &cameraIndices[30]);
	glDrawElements(GL_TRIANGLE_STRIP, 44, GL_UNSIGNED_INT, &cameraIndices[69]);
	glDrawElements(GL_TRIANGLE_STRIP, 44, GL_UNSIGNED_INT, &cameraIndices[113]);
	glDrawElements(GL_TRIANGLE_STRIP, 44, GL_UNSIGNED_INT, &cameraIndices[157]);
	glDrawElements(GL_TRIANGLE_STRIP, 44, GL_UNSIGNED_INT, &cameraIndices[201]);

	glDisableClientState(GL_VERTEX_ARRAY);	// disable vertex arrays
	glDisableClientState(GL_NORMAL_ARRAY);	// disable normal arrays
}

///////////////////////////////////////////////////////////////////////////////
// ��initialize GLUT for windowing��
///////////////////////////////////////////////////////////////////////////////
int initGLUT(int argc, char **argv)
{
    // GLUT stuff for windowing
    // initialization openGL window.
    // it is called before any other GLUT routine
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_STENCIL);   // display mode

    glutInitWindowSize(screenWidth, screenHeight);  // window size

    glutInitWindowPosition(100, 100);           // window location

    // finally, create a window with openGL context
    // Window will not displayed until glutMainLoop() is called
    // it returns a unique ID
    int handle = glutCreateWindow(argv[0]);     // param is the title of window

    // register GLUT callback functions
    glutDisplayFunc(displayCB);
    glutTimerFunc(33, timerCB, 33);             // redraw only every given millisec
    //glutIdleFunc(idleCB);                       // redraw whenever system is idle
    glutReshapeFunc(reshapeCB);
    glutKeyboardFunc(keyboardCB);
    glutMouseFunc(mouseCB);
    glutMotionFunc(mouseMotionCB);

    return handle;
}


///////////////////////////////////////////////////////////////////////////////
// ��initialize OpenGL��
// disable unused features
///////////////////////////////////////////////////////////////////////////////
void initGL()
{
    glShadeModel(GL_SMOOTH);                    // shading mathod: GL_SMOOTH or GL_FLAT
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);      // 4-byte pixel alignment

    // enable /disable features
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_CULL_FACE);

    // enable /disable features
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);

     // track material ambient and diffuse from surface color, call it before glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    glClearColor(0, 0, 0, 0);                   // background color
    glClearStencil(0);                          // clear stencil buffer
    glClearDepth(1.0f);                         // 0 is near, 1 is far
    glDepthFunc(GL_LEQUAL);

    initLights();
}


///////////////////////////////////////////////////////////////////////////////
// write 2d text using GLUT ��д2D���֡�
// The projection matrix must be set to orthogonal before call this function.
///////////////////////////////////////////////////////////////////////////////
void drawString(const char *str, int x, int y, float color[4], void *font)
{
    glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT); // lighting and color mask
    glDisable(GL_LIGHTING);     // need to disable lighting for proper text color
    glDisable(GL_TEXTURE_2D);

    glColor4fv(color);          // set text color
    glRasterPos2i(x, y);        // place text position

    // loop all characters in the string
    while(*str)
    {
        glutBitmapCharacter(font, *str);
        ++str;
    }

    glEnable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);
    glPopAttrib();
}


///////////////////////////////////////////////////////////////////////////////
// ��initialize global variables��
///////////////////////////////////////////////////////////////////////////////
bool initSharedMem()
{
    screenWidth = SCREEN_WIDTH;
    screenHeight = SCREEN_HEIGHT;

    mouseLeftDown = mouseRightDown = false;
    mouseX = mouseY = 0;

    cameraAngleX = cameraAngleY = 0;
    cameraDistance = CAMERA_DISTANCE;

    drawMode = 0; // 0:fill, 1: wireframe, 2:points

    return true;
}


///////////////////////////////////////////////////////////////////////////////
// clean up shared memory
///////////////////////////////////////////////////////////////////////////////
void clearSharedMem()
{
}


///////////////////////////////////////////////////////////////////////////////
// initialize lights ����ʼ�����ա�
///////////////////////////////////////////////////////////////////////////////
void initLights()
{
    // set up light colors (ambient, diffuse, specular)
    GLfloat lightKa[] = {.2f, .2f, .2f, 1.0f};  // ambient light ������
    GLfloat lightKd[] = {.7f, .7f, .7f, 1.0f};  // diffuse light �������
    GLfloat lightKs[] = {1, 1, 1, 1};           // specular light �����
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightKa);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightKd);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightKs);

    // position the light ��Դλ��
    float lightPos[4] = {0, 0, 20, 1}; // positional light
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glEnable(GL_LIGHT0);                        // MUST enable each light source after configuration
}


///////////////////////////////////////////////////////////////////////////////
// display info messages ����ʾ��Ϣ��======================
///////////////////////////////////////////////////////////////////////////////
void showInfo()
{
    // backup current model-view matrix
    glPushMatrix();                     // save current modelview matrix
    glLoadIdentity();                   // reset modelview matrix

    // set to 2D orthogonal projection
    glMatrixMode(GL_PROJECTION);        // switch to projection matrix
    glPushMatrix();                     // save current projection matrix
    glLoadIdentity();                   // reset projection matrix
    gluOrtho2D(0, screenWidth, 0, screenHeight); // set to orthogonal projection

    float color[4] = {1, 1, 1, 1};//��ɫ����

    stringstream ss;
    ss << std::fixed << std::setprecision(3);

    drawString("=== View Matrix ===", 0, screenHeight-TEXT_HEIGHT, color, font);
    ss << "[" << std::setw(8) << matrixView[0] << std::setw(8) << matrixView[4] << std::setw(8) << matrixView[8] << std::setw(8) << matrixView[12] << "]" << ends;
    drawString(ss.str().c_str(), 0, screenHeight-(2*TEXT_HEIGHT), color, font);
    ss.str("");
    ss << "[" << std::setw(8) << matrixView[1] << std::setw(8) << matrixView[5] << std::setw(8) << matrixView[9] << std::setw(8) << matrixView[13] << "]" << ends;
    drawString(ss.str().c_str(), 0, screenHeight-(3*TEXT_HEIGHT), color, font);
    ss.str("");
    ss << "[" << std::setw(8) << matrixView[2] << std::setw(8) << matrixView[6] << std::setw(8) << matrixView[10]<< std::setw(8) << matrixView[14] << "]" << ends;
    drawString(ss.str().c_str(), 0, screenHeight-(4*TEXT_HEIGHT), color, font);
    ss.str("");
    ss << "[" << std::setw(8) << matrixView[3] << std::setw(8) << matrixView[7] << std::setw(8) << matrixView[11]<< std::setw(8) << matrixView[15]<< "]" << ends;
    drawString(ss.str().c_str(), 0, screenHeight-(5*TEXT_HEIGHT), color, font);
    ss.str("");

	drawString("=== position angle ===", 0, 9 * TEXT_HEIGHT, color, font);
	ss << "x" << "[" << std::setw(8) << pos_x << std::setw(8) << agl_x << "]" << ends;
	drawString(ss.str().c_str(), 0, 8 * TEXT_HEIGHT, color, font);
	ss.str("");
	ss << "y" << "[" << std::setw(8) << pos_y << std::setw(8) << agl_y << "]" << ends;
	drawString(ss.str().c_str(), 0, 7 * TEXT_HEIGHT, color, font);
	ss.str("");
	ss << "z" << "[" << std::setw(8) << pos_z << std::setw(8) << agl_z << "]" << ends;
	drawString(ss.str().c_str(), 0, 6 * TEXT_HEIGHT, color, font);
	ss.str("");

    drawString("=== Model Matrix ===", 0, 4*TEXT_HEIGHT, color, font);
    ss << "[" << std::setw(8) << matrixModel[0] << std::setw(8) << matrixModel[4] << std::setw(8) << matrixModel[8] << std::setw(8) << matrixModel[12] << "]" << ends;
    drawString(ss.str().c_str(), 0, 3*TEXT_HEIGHT, color, font);
    ss.str("");
    ss << "[" << std::setw(8) << matrixModel[1] << std::setw(8) << matrixModel[5] << std::setw(8) << matrixModel[9] << std::setw(8) << matrixModel[13] << "]" << ends;
    drawString(ss.str().c_str(), 0, 2*TEXT_HEIGHT, color, font);
    ss.str("");
    ss << "[" << std::setw(8) << matrixModel[2] << std::setw(8) << matrixModel[6] << std::setw(8) << matrixModel[10]<< std::setw(8) << matrixModel[14] << "]" << ends;
    drawString(ss.str().c_str(), 0, TEXT_HEIGHT, color, font);
    ss.str("");
    ss << "[" << std::setw(8) << matrixModel[3] << std::setw(8) << matrixModel[7] << std::setw(8) << matrixModel[11]<< std::setw(8) << matrixModel[15] << "]" << ends;
    drawString(ss.str().c_str(), 0, 0, color, font);
    ss.str("");

    // unset floating format
    ss << std::resetiosflags(std::ios_base::fixed | std::ios_base::floatfield);

    // restore projection matrix
    glPopMatrix();                   // restore to previous projection matrix

    // restore modelview matrix
    glMatrixMode(GL_MODELVIEW);      // switch to modelview matrix
    glPopMatrix();                   // restore to previous modelview matrix
}


///////////////////////////////////////////////////////////////////////////////
// ������һ��͸��ƽ��ͷ�塿
// set a perspective frustum with 6 params similar to glFrustum()
// (left, right, bottom, top, near, far)
///////////////////////////////////////////////////////////////////////////////
Matrix4 setFrustum(float l, float r, float b, float t, float n, float f)
{
    Matrix4 mat;
    mat[0]  =  2 * n / (r - l);
    mat[5]  =  2 * n / (t - b);
    mat[8]  =  (r + l) / (r - l);
    mat[9]  =  (t + b) / (t - b);
    mat[10] = -(f + n) / (f - n);
    mat[11] = -1;
    mat[14] = -(2 * f * n) / (f - n);
    mat[15] =  0;
    return mat;
}



///////////////////////////////////////////////////////////////////////////////
// ������һ���Գ�͸��ƽ��ͷ�塿
// set a symmetric perspective frustum with 4 params similar to gluPerspective
// (vertical field of view, aspect ratio, near, far)
///////////////////////////////////////////////////////////////////////////////
Matrix4 setFrustum(float fovY, float aspectRatio, float front, float back)
{
    float tangent = tanf(fovY/2 * DEG2RAD);   // tangent of half fovY
    float height = front * tangent;           // half height of near plane
    float width = height * aspectRatio;       // half width of near plane

    // params: left, right, bottom, top, near, far
    return setFrustum(-width, width, -height, height, front, back);
}



//=============================================================================
// CALLBACKS ���ص�������
//=============================================================================

//����������������������ʾ�ص���������������������������
void displayCB()
{
	//======
	proImg();

    // clear buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // save the initial ModelView matrix before modifying ModelView matrix
    glPushMatrix();

    // tramsform camera =========���۲����=========
    matrixView.identity();
    matrixView.rotate(cameraAngleY, 0, 1, 0);//��ת
    matrixView.rotate(cameraAngleX, 1, 0, 0);//��ת
    matrixView.translate(0, 0, -cameraDistance);//ƽ�� ��ʼ6
    //@@ the equivalent code for using OpenGL routine is:
    //@@ glTranslatef(0, 0, -cameraDistance);
    //@@ glRotatef(cameraAngleX, 1, 0, 0);   // pitch
    //@@ glRotatef(cameraAngleY, 0, 1, 0);   // heading

    // copy view matrix to OpenGL
    glLoadMatrixf(matrixView.get());

    drawGrid();        // ������� draw XZ-grid with default size
	drawGridXY();

    // compute model matrix ����=========��ģ�;���==========
    matrixModel.identity();
    matrixModel.rotateZ(agl_z);        // rotate 45 degree on Z-axis
    matrixModel.rotateY(agl_y);        // rotate 45 degree on Y-axis ��Y��ת45��
    matrixModel.rotateX(agl_x);        // rotate 45 degree on X-axis
    matrixModel.translate(pos_x, pos_y, pos_z); // move 2 unit up

    // compute modelview matrix ���㡾ģ�͹۲����
    matrixModelView = matrixView * matrixModel;

    // copy modelview matrix to OpenGL
    glLoadMatrixf(matrixModelView.get());

    drawAxis();//����
	//======תһ�����
	matrixModel.identity();
	matrixModel.rotateX(180);
	matrixModel.translate(pos_x, pos_y, pos_z);
	matrixModelView = matrixView * matrixModel;
	glLoadMatrixf(matrixModelView.get());
	//======
	drawCamera();//drawModel();//��ģ��

    // draw info messages
    showInfo();//��ʾ��Ϣ

    glPopMatrix();

    glutSwapBuffers();
}


//��������С�ص�������
void reshapeCB(int w, int h)
{
    screenWidth = w;
    screenHeight = h;

    // set viewport to be the entire window
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);

    // set perspective viewing frustum
    glMatrixMode(GL_PROJECTION);
    matrixProjection = setFrustum(45, (float)w/h, 1.0f, 100.0f);//ƽ��ͷ��
    glLoadMatrixf(matrixProjection.get());
    //@@ the equivalent OpenGL call
    //@@ gluPerspective(45.0f, (float)(w)/h, 1.0f, 100.0f); // FOV, AspectRatio, NearClip, FarClip

    // DEBUG
    std::cout << "===== Projection Matrix =====\n";
    std::cout << matrixProjection << std::endl;

    // switch to modelview matrix in order to set scene
    glMatrixMode(GL_MODELVIEW);
}

//ʱ��ص�����
void timerCB(int millisec)
{
    glutTimerFunc(millisec, timerCB, millisec);
    glutPostRedisplay();
}

//idle?
void idleCB()
{
    glutPostRedisplay();
}

//=================�����̻ص�������==================
void keyboardCB(unsigned char key, int x, int y)
{
    switch(key)
    {
    case 27: // ESCAPE
        exit(0);
        break;

    case ' ':
        break;

    case 'd': // switch rendering modes (fill -> wire -> point)
    case 'D':
        drawMode = ++drawMode % 3;
        if(drawMode == 0)        // fill mode
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);//���
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_CULL_FACE);
        }
        else if(drawMode == 1)  // wireframe mode
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//��
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
        }
        else                    // point mode
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);//��
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
        }
        break;
	case 'q': pos_x += 1; break;
	case 'a': pos_y += 1; break;
	case 'z': pos_z += 1; break;
	case 'w': pos_x -= 1; break;
	case 's': pos_y -= 1; break;
	case 'x': pos_z -= 1; break;

	case 'r': agl_x += 1; break;
	case 'f': agl_y += 1; break;
	case 'v': agl_z += 1; break;
	case 't': agl_x -= 1; break;
	case 'g': agl_y -= 1; break;
	case 'b': agl_z -= 1; break;
    default:
        ;
    }
}

//����갴���ص�������
void mouseCB(int button, int state, int x, int y)
{
    mouseX = x;
    mouseY = y;

    if(button == GLUT_LEFT_BUTTON)//���
    {
        if(state == GLUT_DOWN)
        {
            mouseLeftDown = true;
        }
        else if(state == GLUT_UP)
            mouseLeftDown = false;
    }

    else if(button == GLUT_RIGHT_BUTTON)//�Ҽ�
    {
        if(state == GLUT_DOWN)
        {
            mouseRightDown = true;
        }
        else if(state == GLUT_UP)
            mouseRightDown = false;
    }
}

//������ƶ��ص�������
void mouseMotionCB(int x, int y)
{
    if(mouseLeftDown)//���
    {
        cameraAngleY -= (x - mouseX);//+=
        cameraAngleX += (y - mouseY);
        mouseX = x;
        mouseY = y;
    }
    if(mouseRightDown)//�Ҽ�
    {
        cameraDistance += (y - mouseY) * 0.2f;
        mouseY = y;
    }
}

//�˳��ص�����
void exitCB()
{
    clearSharedMem();
}

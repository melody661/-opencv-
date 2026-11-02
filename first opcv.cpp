//喵喵喵隶属于gmh
//问题：遇到其他地方多了的光晕会多出一个 区块而让我误判，导致坐标错位
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
using namespace cv;
using namespace std;
struct zuobiao
{
    double x;//左上角坐标
    double y;//左上角坐标
    double width;
    double hight; 
}A[1000];
int main(int argc, char** argv) {


    string path ="red_tilted_left.png";//读图
    //blue_straight_ahead.png | blue_tilted_left.png | blue_tilted_right.png | red_straight_ahead.png | red_tilted_left.png | red_tilted_right.png

    Mat img = imread(path);
    if (img.empty())  { cout << "读不到图片" << endl; return -1; }

    // ---------- 1. 用“高饱和度”把灯条芯部和光晕分开 ----------
    Mat hsv; 
    cvtColor(img, hsv, COLOR_BGR2HSV);//转换图片类型 hsv是最常用的抠图格式？
    Mat mask;
    // S >= 110 只留鲜艳的颜色；光晕/过曝白芯接近白色(S 很低)，在这里被滤掉
    inRange(hsv, Scalar(0, 110, 230), Scalar(255, 255, 255), mask);//在hsv空间中，扣出指定颜色区域，左s颜色下限，右s上限
    //mark是被扣出来的区域，白色是被扣出来的，黑色是被滤掉的


    // 这部分我要把灯条中因为光晕而被滤掉的部分补回
    // 用一次小核闭运算把它补上，灯条就变回实心长条。
    morphologyEx(mask, mask, MORPH_CLOSE,                   //闭运算,这个用来处理图像的噪点，闭运算是用来填充断裂区的
     getStructuringElement(MORPH_RECT, Size(15, 15)));//size越大处越强，MORPH_ELLIPSE是椭圆形的，cross是 十字的，rect矩形
    morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(6,6)));//DSH建议再加开运算，用于腐蚀小噪点！！
    // ---------- 2. 找连通域 ----------
    vector<vector<Point>> contours;//轮廓容器，用于存储轮廓点集
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//找轮廓，RETR_EXTERNAL只找最外层轮廓，CHAIN_APPROX_SIMPLE压缩轮廓点集
    Mat out = img.clone();//克隆原图，用于画
    int count = 0;

    for (size_t i = 0; i < contours.size(); ++i) {//size_t是无符号整型，long long是有符号整型，size_t是平台相关的类型，通常用于表示对象的大小或索引。size_t的大小取决于平台的位数，在32位系统上通常为4字节，在64位系统上通常为8字节。long long在C++标准中保证至少为8字节，但它是有符号的，可以表示负数。使用size_t可以确保在不同平台上代码的一致性和可移植性，尤其是在处理容器大小和索引时。
        Rect box = boundingRect(contours[i]);// 描述轮廓的外接矩形，返回一个Rect对象，包含x,y,width,height
        //tl()是左上角坐标，br()是右下角坐标，size()是宽高，area()是面积
        if (box.area() < 600) continue;                 // 小噪点不处理
        if (box.width < 10) continue;                   // 太细的条纹（桌面反光）不处理
        if (box.height < 1.1 * box.width) continue;    // 灯条是竖长条（放宽，允许光晕把框撑胖）不符合比例不处理
        //判断红蓝
        Mat one = Mat::zeros(img.size(), CV_8U);//构建全0图，用于画轮廓，cv-8u是8位无符号单通道图像 8uc1是单通道灰度图，c3是三色彩图
        drawContours(one, contours, (int)i, Scalar(255), FILLED);//把contours扣的东西显示在one上，i指定第几个轮廓，Scalar(255)是白色，FILLED是填充轮廓
        Scalar BGR = mean(img, one);//求平均,用主体去掩盖白光带来的误差
        double B = BGR[0], G = BGR[1], R = BGR[2];
        bool wtcolor = (R > B);                           
        string name = wtcolor ? "红色" : "蓝色";//1左0右
        Scalar color = wtcolor ? Scalar(0, 0, 255) : Scalar(255, 0, 0);  // BGR
        //直接三项运算判定颜色，不用改参数

        // 坐标
        A[i].x = box.x; A[i].y = box.y;A[i].hight=box.height;A[i].width=box.width;
        Point2f center(box.x + box.width / 2.0f, box.y + box.height / 2.0f);//point(int)f是浮点型坐标，point是整型坐标
 //       printf("%-6d %-6s (%d, %d, %d, %d)      (%.0f, %.0f)     (%.0f, %.0f, %.0f)\n",
 //              ++count, name, box.x, box.y, box.width, box.height,
  //             center.x, center.y, B, G, R);

    //   imshow("test", out);
     //   waitKey(3000);

        Mat overlay = out.clone();//改进点，为了使得画的东西像是在原图上，而不是覆盖原图
       drawContours(overlay, contours, (int)i, color, FILLED);//在参数1中将参数2对应color的地方画上线条
        addWeighted(overlay, 0.70, out, 0.30, 0, out);// 70% 覆盖，30% 原图,要一样大小的图

     //   imshow("test", overlay);
     //   waitKey(3000);

        rectangle(out, box, Scalar(0, 255, 0), -1);            //画框
    }
    double x1=0,y1=0;
    cout<<contours.size()<<endl;
    int k=0; double maxx=0;
    for(int i=0;i<contours.size();i++) maxx=max(maxx,contourArea(contours[i]));
    for(int i=0;i<contours.size();i++)
    {
        if(contourArea(contours[i])*4<maxx) continue;                 // 小噪点不处理
        k++;
        if(A[i].width>10&&A[i].hight>1.1*A[i].width)
        {
            x1+=A[i].x+A[i].width/2;
            y1+=A[i].y+A[i].hight/2;
        }
    }
    cout<<k<<endl;
    x1/=k;
    y1/=k;
    circle(out, Point(x1,y1), 10, Scalar(0, 255, 0), -1);
    imwrite(path+"_result.png", out);
 //   imwrite("out_mask.png", mask);

    namedWindow("mimicat", WINDOW_FREERATIO);
    namedWindow("catmimi", WINDOW_FREERATIO);
    imshow("mimicat", out);
    imshow("catmimi", mask);
    waitKey(0);
    return 0;
}

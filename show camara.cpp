//隶属于gmh的二代目喵，改变：基于一代目（把它塞进了函数进行调用），然后略改参数使其可以适应视频
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
using namespace cv;
using namespace std;

// ============================ 可改的参数 ============================
static const int CAMERA_INDEX = 0;      // 摄像头编号
static const int FRAME_WIDTH = 640;     // 相机分辨率（龙邱一般是 640x480）可以960*
static const int FRAME_HEIGHT = 480;    
static const int FRAME_FPS = 144;        //  理想帧
static const int BRIGHT=10;            //亮度  -64-64 环境强光则参数调低，反之调高
static const int BAO=40;
// ===================================================================
Mat frame, out, mask;
struct zuobiao {
    double x;       // 左上角坐标
    double y;       // 左上角坐标
    double width;
    double hight;
} A[1000];

static void Miao(const Mat& img, Mat& out, Mat& mask) {
    // ---------- 1. 用“高饱和度”把灯条芯部和光晕分开 ----------
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);      // 转换图片类型，hsv 是最常用的抠图格式

    // S >= 110 只留鲜艳的颜色；光晕/过曝白芯接近白色(S 很低)，在这里被滤掉, 这是为了扣出明显的红、蓝光
    inRange(hsv, Scalar(0, 100, 180), Scalar(255, 255, 255), mask);
    // mask 是被扣出来的区域，白色是被扣出来的，黑色是被滤掉的

    // 这部分要把灯条中因为光晕而被滤掉的部分补回
    // 用一次小核闭运算把它补上，灯条就变回实心长条
    morphologyEx(mask, mask, MORPH_CLOSE,
                 getStructuringElement(MORPH_RECT, Size(15, 15)));
    // 再加一次开运算，腐蚀掉孤立的小噪点
    morphologyEx(mask, mask, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));

    // ---------- 2. 找连通域 ----------
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//contours是各部分轮廓的存储数组

    out = img.clone();                      // 克隆原图，用于画

    // 摄像头对着别处时一帧里可能一个轮廓都没有；原代码会因为除以 0 得到 NaN
    // （圆画到 (NaN,NaN)）。这里直接原图返回。这是 相比 电脑识图的改进处,当然，被干扰了那无可避免
    if (contours.empty()) return;

    for (size_t i = 0; i < contours.size(); ++i) {
        Rect box = boundingRect(contours[i]);
        if (box.area() < 400) continue;                 // 小噪点不处理
        if (box.width < 8) continue;                   // 太细的条纹（桌面反光）不处理
        if (box.height < 1.1 * box.width) continue;     // 灯条是竖长条

        // 判断红蓝
        Mat one = Mat::zeros(img.size(), CV_8U); // 生成一个全黑的掩码图像
        drawContours(one, contours, (int)i, Scalar(255), FILLED);// 用轮廓填充掩码图像
        Scalar BGR = mean(img, one);                    // 求平均，用主体掩盖白光带来的误差
        double B = BGR[0], G = BGR[1], R = BGR[2];
        bool wtcolor = (R > B);//经典的投票判定红蓝
        string name = wtcolor ? "红色" : "蓝色";
        Scalar color = wtcolor ? Scalar(0, 0, 255) : Scalar(255, 0, 0);   // BGR

        // 坐标
        A[i].x = box.x;
        A[i].y = box.y;
        A[i].hight = box.height;
        A[i].width = box.width;

        Mat overlay = out.clone();                      // 让涂色像盖在原图上，而不是覆盖
        drawContours(overlay, contours, (int)i, color, FILLED);// 用轮廓填充掩码图像
        addWeighted(overlay, 0.70, out, 0.30, 0, out);  // 70% 覆盖，30% 原图
        rectangle(out, box, Scalar(0, 255, 0), -1);     // 画框（原代码就是实心绿框）
    }

    // ---------- 3. 平均坐标 ----------
    double x1 = 0, y1 = 0;
    int k = 0;
    double maxx = 0;
    for (int i = 0; i < (int)contours.size(); i++) maxx = max(maxx, contourArea(contours[i]));
    //求最大的图块，用于去除小区域干扰
    for (int i = 0; i < (int)contours.size(); i++) {
        if (contourArea(contours[i]) * 6 < maxx or maxx<1000) continue;   // 小噪点不处理 重要去除噪点的参数
        if (A[i].width > 10 && A[i].hight > 1.1 * A[i].width) {
            x1 += A[i].x + A[i].width / 2;
            y1 += A[i].y + A[i].hight / 2;
            ++k;
        }
    }
    x1 /=k;
    y1 /=k;
    if(k>=2) circle(out, Point(x1, y1), 10, Scalar(0, 255, 0), -1);//防止准心乱闪
}

int main(int argc, char** argv) {
    VideoCapture cap;
    bool op=cap.isOpened();
    if(!op)
    {
        cout<<"没开摄像头！！！" <<endl;
        return 0;
    }

    //传参
    cap.set(CAP_PROP_AUTO_EXPOSURE,0.25);//关闭自动曝光
    cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);//宽、长参数
    cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);//
    cap.set(CAP_PROP_FPS, FRAME_FPS);//fps设定
    cap.set(CAP_PROP_BUFFERSIZE, 1);//缓冲帧，竞赛设1？
    cap.set(CAP_PROP_BRIGHTNESS,BRIGHT);//亮度
    cap.set(CAP_PROP_EXPOSURE,BAO);//曝光
    //摄像头参数我扔开头宏定义了

    while(true) {
        if (!cap.read(frame) || frame.empty()) {// 读摄像头失败
            cout<<"manba out"<<endl;
            break;
        }
        Miao(frame, out, mask);
        namedWindow("fact",WINDOW_FREERATIO);
        imshow("fact", out);
        imshow("test", mask);
        // 懒得搞按键了。esc退出得了 
    }
    cap.release();// 释放摄像头好习惯
    destroyAllWindows();// 关闭窗口好习惯
    return 0;
}

//隶属于gmh的二代目喵，改变：基于一代目（把它塞进了函数进行调用），然后略改参数使其可以适应视频
//根据视频跟踪的分析：要改的：
//1.灯条长宽判断在倒转时不成立 √
//2.将红蓝一起判断变为要么判断红要么判断蓝，这样的精度应该会提升？√不过我发现精度反而下降了？
//3.写一个留帧的设计，用于识别不出时候可以视觉上进行掩饰 √
//4.保留最大的两个矩形的做法似乎有很多毛病？（终止）

//4.0优化：对于矩形肥大的优化。
//我决定在识别的基础上缩小矩形 
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
using namespace cv;
using namespace std;
//static 全局静态变量,限制作用区仅为当前文件
// ============================ 可改的参数 ============================
static const int CAMERA_INDEX = 0;      // 摄像头编号
static const int FRAME_WIDTH = 640;     // 相机分辨率（龙邱一般是 640x480）
static const int FRAME_HEIGHT = 480;    
static const int FRAME_FPS = 60;        //  理想帧 龙湫60或30
static const int BRIGHT=200;            //亮度  0-500 环境强光则参数调低，反之调高
static const int BAO=-6;                //曝光    
// ===================================================================
Mat frame, out, mask;
struct zuobiao {
    double x;       // 左上角坐标
    double y;       // 左上角坐标
    double width;
    double hight;
} A[1000];
static void Miao(const Mat& img, Mat& out, Mat& mask) {
    Mat hsv;//用hsv分参
    cvtColor(img, hsv, COLOR_BGR2HSV);      // 转换图片类型，hsv 是最常用的抠图格式,因为可以把亮度饱和度和色彩分开     

    // S >= 110 只留鲜艳的颜色；光晕/过曝白芯接近白色(S 很低)，在这里被滤掉, 这是为了扣出明显的红、蓝光
   // inRange(hsv, Scalar(0,130,150), Scalar(50,255,255), mask);//红色    左下限右上限
   inRange(hsv,Scalar(100,130,80),Scalar(255,255,255),mask);//蓝色 

    //一般情况下：REDD(0,130,150 )   REDU(15,255,255)  BLUED(100,130,80)  BLUEU(255,255,255)
    //过亮：R(0,160,110)-(15,255,255)                B(100,160,110)-(125,255,255)
    //暗R(0,110,55)-(15,255,255)                    B(100,110,55)-(125,255,255)




    // 这部分要把灯条中因为光晕而被滤掉的部分补回
    // 用一次小核闭运算把它补上，灯条就变回实心长条,闭合;先扩张连一块再消除边缘.
    morphologyEx(mask, mask, MORPH_CLOSE,
                 getStructuringElement(MORPH_RECT, Size(10,10)));
    // 再加一次开运算，腐蚀掉孤立的小噪点
    morphologyEx(mask, mask, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));//size是处理的强度

    // 存矩形
    vector<vector<Point>> contours;//御用矩形存储形式
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//contours是各部分轮廓的存储数组
    out = img.clone();                      // 克隆原图，用于画
    for (size_t i = 0; i < contours.size(); ++i) {
        Rect box = boundingRect(contours[i]);//拿出每个矩形
    if (box.area() < 400) continue;                 // 小噪点不处理
     if (box.width < 6 or box.height < 6) continue;                   // 太细的条纹（桌面反光）不处理
      // if (box.height < 1.1 * box.width) continue;     // 灯条是竖长条(倒置识别不出的原因？
        A[i].x = box.x;
        A[i].y = box.y;
        A[i].hight = box.height;
        A[i].width = box.width;
    }
    double x1 = 0, y1 = 0;
    int k = 0;
    double maxx = 0;
    //取俩最大矩形法
 /* double secmaxx=0;
    int FIS=0,SEC=0;
    for (int i = 0; i < (int)contours.size(); i++){
        if (contourArea(contours[i]) > maxx)
        {
            secmaxx=maxx,SEC=FIS;
            maxx=contourArea(contours[i]),FIS=i;
        }
        else if(contourArea(contours[i]) > secmaxx)
        {
            secmaxx=contourArea(contours[i]) ,SEC=i;
        }
    }   
    x1=(A[FIS].x+A[FIS].width/2+A[SEC].x+A[SEC].width/2)/2;
    y1=(A[FIS].y+A[FIS].hight/2+A[SEC].y+A[SEC].hight/2)/2;
    if(!FIS and !SEC)
    {
        circle(out, Point(x1, y1), 10, Scalar(0, 255, 0), -1);//防止准心乱闪
    } */
   int maxp;
   for(int  i=0;i<(int)contours.size();i++)
   {
    if(contourArea(contours[i])>maxx)
         maxx=max(maxx,contourArea(contours[i])),maxp=i;
   }
   for (int i = 0; i < (int)contours.size(); i++) 
   {
       if (contourArea(contours[i]) * 6 < maxx) continue;   // 小噪点不处理 重要去除噪点的参数
       Rect box=boundingRect(contours[i]);
       Rect conp=boundingRect(contours[maxp]);
       if((abs(box.x-conp.x)>100 or abs(box.y-conp.y)>150)&& maxx<1000)continue;//4.0新增判定喵
       if(A[i].hight*10<A[i].width || A[i].hight>A[i].width*10)continue;//你敢信我只是想把进度条优化了？？
        if (A[i].width >5) {
           x1 += A[i].x + A[i].width / 2;
            y1 += A[i].y + A[i].hight / 2;
           ++k;
        }
        
    rectangle(out, boundingRect(contours[i]), Scalar(0, 255, 0), 4);//在这画，优化视觉体验喵
   }

    x1 /=k;
    y1 /=k; 
    if(k>=2)
    {
        circle(out, Point(x1, y1), 10, Scalar(0, 255, 0), -1);//防止准心乱闪
        waitKey(2);//留帧设计
    } 


}

int main(int argc, char** argv) {  
    VideoCapture cap;
    cap.open(CAMERA_INDEX);          // open 摄像头
    bool op=cap.isOpened();
    if(!op)
    {
        cout<<"没开摄像头！！！" <<endl;
        return 0;
    }

    //传参
    cap.set(CAP_PROP_AUTO_EXPOSURE,0);//关闭自动曝光 0关1开  
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
        namedWindow("test",WINDOW_FREERATIO);
        imshow("fact", out);
        imshow("test", mask);
        if (waitKey(1) == 27) break;
        // 懒得搞按键了。esc退出得了 
    }
    cap.release();// 释放摄像头好习惯
    destroyAllWindows();// 关闭窗口好习惯
    return 0;
}

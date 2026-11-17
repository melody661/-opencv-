//隶属于gmh的二代目喵，改变：基于一代目（把它塞进了函数进行调用），然后略改参数使其可以适应视频
//根据视频跟踪的分析：要改的：
//1.灯条长宽判断在倒转时不成立 √
//2.将红蓝一起判断变为要么判断红要么判断蓝，这样的精度应该会提升？√不过我发现精度反而下降了？
//3.写一个留帧的设计，用于识别不出时候可以视觉上进行掩饰 √
//4.保留最大的两个矩形的做法似乎有很多毛病？（终止）
//4.0优化：对于矩形肥大的优化。
//我决定在识别的基础上缩小矩形 
//5.0:采取曝光最低、用hsv调制参数分离白光和假色以及暗影子的方式，效果非常好。
//5.1优化：决定加入滑动条来提高调试参数的效率      
//5.2优化：决定使用旋转矩形来更精准采集四点坐标。
//6.0: 根据老登提示：决定用平行特性加循环筛选来选择矩形，达到可以识别多个合法装甲板的目的，并希望可以进行分类，以便于后续的预判设计           
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
static const int BRIGHT=100;            //亮度  0-500 环境强光则参数调低，反之调高
static const int BAO=-10;                //曝光    
// ===================================================================
Mat hdt;
Mat frame, out, mask;
int Hmin=90,Smin=50,Vmin=90,Hmax=130,Smax=255,Vmax=255;  
// 蓝色参数：（90，50，90）--（130，255，255）
// inRange(hsv, Scalar(170,70,90), Scalar(179,255,255), mask);
//  inRange(hsv, Scalar(0,70,90), Scalar(10,255,255), mask);//红色（俩个一起开）    左下限右上限
static void onHSVTrackbar(int, void*)
{
    char title[160];
    snprintf(title, sizeof(title), "HSV  H:%d-%d  S:%d-%d  V:%d-%d",
             Hmin, Hmax, Smin, Smax, Vmin, Vmax);
    setWindowTitle("huadongtiao", title);//只改标题，不动画面
}

static void createHSVTrackbars()
{
    namedWindow("huadongtiao", WINDOW_NORMAL);
    resizeWindow("huadongtiao", 420, 260);

    // H 色相：OpenCV 里 H 最大是 179
    createTrackbar("Hmin", "huadongtiao", &Hmin, 179, onHSVTrackbar);
    createTrackbar("Hmax", "huadongtiao", &Hmax, 179, onHSVTrackbar);
    // S 饱和度 / V 亮度：0-255
    createTrackbar("Smin", "huadongtiao", &Smin, 255, onHSVTrackbar);
    createTrackbar("Smax", "huadongtiao", &Smax, 255, onHSVTrackbar);
    createTrackbar("Vmin", "huadongtiao", &Vmin, 255, onHSVTrackbar);
    createTrackbar("Vmax", "huadongtiao", &Vmax, 255, onHSVTrackbar);
}
// =====================================================================

struct zuobiao {
    double x;       // 左上角坐标
    double y;       // 左上角坐标
    double width;
    double hight;
} A[1000];
static void Miao(const Mat& img, Mat& out, Mat& mask) {
    Mat hsv;//用hsv分参
    cvtColor(img, hsv, COLOR_BGR2HSV);      // 转换图片类型，hsv 是最常用的抠图格式,因为可以把亮度饱和度和色彩分开   
    //H 90-130是蓝色，调大往青蓝绿走，调小往紫红走    红色0-10橘红，170-179品红深红
    //s 饱和度可以拿来区分纯色彩 和参杂白光的伪色彩
    //v 亮度，v的下限决定能否过滤暗处


   inRange(hsv,Scalar(Hmin,Smin,Vmin),Scalar(Hmax,Smax,Vmax),mask);//蓝色 

    //开红色俩个都要开因为红色是0-10加上170-179
  // inRange(hsv,Scalar(Hmin,Smin,Vmin),Scalar(Hmax,Smax,Vmax),mask);
   //inRange(hsv,Scalar(0,Smin,Vmin),Scalar(10,Smax,Vmax),mask);


    // 这部分要把灯条中因为光晕而被滤掉的部分补回
    // 用一次小核闭运算把它补上，灯条就变回实心长条,闭合;先扩张连一块再消除边缘.
    morphologyEx(mask, mask, MORPH_CLOSE,
                 getStructuringElement(MORPH_RECT, Size(25,25)));
    // 再加一次开运算，腐蚀掉孤立的小噪点
    morphologyEx(mask, mask, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));//size是处理的强度
    // 存矩形
    vector<vector<Point>> contours;//御用矩形存储形式
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//contours是各部分轮廓的存储数组，这个函数是把存储的点阵画到mask上
    out = img.clone();                      // 克隆原图，用于画
    for (size_t i = 0; i < contours.size(); ++i) {
        //minarearect:旋转矩形！！
        RotatedRect box=minAreaRect(contours[i]);
     /*   Rect box = boundingRect(contours[i]);//拿出每个矩形
        A[i].x = box.x;
        A[i].y = box.y;
        A[i].hight = box.height;
        A[i].width = box.width;
        */
    }
    double x1 = 0, y1 = 0;
    int k = 0;
    double maxx = 0;
   int maxp=0;
   for(int  i=0;i<(int)contours.size();i++)
   {
    if(contourArea(contours[i])>maxx)
         maxx=max(maxx,contourArea(contours[i])),maxp=i;
   }
   for (int i = 0; i < (int)contours.size(); i++) 
   {
    if (contourArea(contours[i]) * 100 < maxx) continue;
    RotatedRect box=minAreaRect(contours[i]);
    x1+=box.center.x;
    y1+=box.center.y;
    k++;
    /*  if (contourArea(contours[i]) * 100 < maxx) continue;   // 小噪点不处理 重要去除噪点的参数
       Rect box=boundingRect(contours[i]);
       Rect conp=boundingRect(contours[maxp]);
       if((abs(box.x-conp.x)>200 or abs(box.y-conp.y)>200)&& maxx<500)continue;//4.0新增判定喵
   //    if(A[i].hight*20<A[i].width || A[i].hight>A[i].width*20)continue;//你敢信我只是想把进度条优化
           x1 += A[i].x + A[i].width / 2;
            y1 += A[i].y + A[i].hight / 2;
           ++k;  */
    drawContours(out, contours,i, Scalar(0, 255, 0), 4);//在这画，优化视觉体验喵
    // 第二个是表示绘画的对象是哪个组，第三个是画 这个组里的第几个？-1是全画。
    //rectangle不和旋转矩形匹配！
   }

    x1 /=k;
    y1 /=k; 
    if(k>=2)
    {
        circle(out, Point(x1, y1), 10, Scalar(0, 255, 0), -1);//防止准心乱闪
        waitKey(1);//留帧设计
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


    createHSVTrackbars();//创建滑动条 


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
        // esc退出 
    }
    cap.release();// 释放摄像头好习惯
    destroyAllWindows();// 关闭窗口好习惯
    return 0;
}

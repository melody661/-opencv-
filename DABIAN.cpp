// 隶属于gmh的二代目喵，改变：基于一代目（把它塞进了函数进行调用），然后略改参数使其可以适应视频
// 根据视频跟踪的分析：要改的：
// 1.灯条长宽判断在倒转时不成立 √
// 2.将红蓝一起判断变为要么判断红要么判断蓝，这样的精度应该会提升？√不过我发现精度反而下降了？
// 3.写一个留帧的设计，用于识别不出时候可以视觉上进行掩饰 √
// 4.保留最大的两个矩形的做法似乎有很多毛病？（终止）
// 4.0优化：对于矩形肥大的优化。
// 我决定在识别的基础上缩小矩形
// 5.0:采取曝光最低、用hsv调制参数分离白光和假色以及暗影子的方式，效果非常好。
// 5.1优化：决定加入滑动条来提高调试参数的效率
// 5.2优化：决定使用旋转矩形来更精准采集四点坐标。
// 6.0: 根据老登提示：决定用平行特性加循环筛选来选择矩形，达到可以识别多个合法装甲板的目的，并希望可以进行分类，以便于后续的预判设计
// 6.0开发日志：11.18，完成了旋转矩形和四个点的坐标读取，差帅选矩形对和数字分类。
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
using namespace cv;
using namespace std;

static const int CAMERA_INDEX = 0;  // 摄像头编号
static const int FRAME_WIDTH = 640; // 相机分辨率（龙邱一般是 640x480）
static const int FRAME_HEIGHT = 480;
static const int FRAME_FPS = 60; //  理想帧 龙湫60或30
static const int BRIGHT = 100;   // 亮度  0-500 环境强光则参数调低，反之调高
static const int BAO = -10;      // 曝光

static const double Lenmin = 20.0;            // 灯条长边最短像素
static const double Changdubi = 0.60;              // 两条灯条长度比的下限
static const double SIN = 0.3422;          // sin(20°) 
static const double COS= 0.40;               // 中心连线与长边夹角的 |cos|
static const double BILI_MIN = 0.6;              // 间距灯条比下限
static const double BILI_MAX = 2.1;              // 间距灯条比上限
//奶包查到的小装甲比例加误差分析后的合理是0.7-1.3，大装甲是1.6到2.4
//实际小装甲1：1   大撞击 2：1
Mat hdt;
Mat frame, out, mask;
struct RecTT
{
    int A, B;
} R[100000];
Point2f Pts[1000][4]; // 拿来存旋转矩阵的四个点;，根据要求分别是0左上1左下2右下3右上
int Hmin = 90, Smin = 160, Vmin = 40, Hmax = 130, Smax = 255, Vmax = 255;
// 蓝色参数：（90，90，100）--（130，255，255）
// inRange(hsv, Scalar(170,70,90), Scalar(179,255,255), mask);
//  inRange(hsv, Scalar(0,70,90), Scalar(10,255,255), mask);//红色（俩个一起开）    左下限右上限
static void onHSVTrackbar(int, void *)
{
    char title[160];
    snprintf(title, sizeof(title), "HSV  H:%d-%d  S:%d-%d  V:%d-%d",
             Hmin, Hmax, Smin, Smax, Vmin, Vmax);
    setWindowTitle("huadongtiao", title); 
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
static void GYH(const RotatedRect &r, Point2f pt[4])
{
    Point2f q[4];
    r.points(q);
    Point2f c(0, 0);
    for (int k = 0; k < 4; k++)
        c += q[k];
    c *= 0.25f;
    for (int a = 0; a < 4; a++)
        for (int b = a + 1; b < 4; b++)
            if (atan2(q[b].y - c.y, q[b].x - c.x) < atan2(q[a].y - c.y, q[a].x - c.x))
                swap(q[a], q[b]);
    int tl = 0;
    for (int k = 1; k < 4; k++)
        if (q[k].x < q[tl].x || (q[k].x == q[tl].x && q[k].y < q[tl].y))
            tl = k;
    for (int k = 0; k < 4; k++)
        pt[k] = q[(tl - k + 4) % 4];
}

struct zuobiao
{
    double x; // 左上角坐标
    double y; // 左上角坐标
    double width;
    double hight;
} A[1000];
static double BILI(const Point2f a[4], const Point2f b[4])
{
    Point2f ea(0, 0), eb(0, 0);
    double La = 0, Lb = 0;
    for (int k = 0; k < 4; k++)
    {
        Point2f v = a[(k + 1) % 4] - a[k];
        double L = norm(v);
        if (L > La)
        {
            La = L;
            ea = v;
        }
    }
    for (int k = 0; k < 4; k++)
    {
        Point2f v = b[(k + 1) % 4] - b[k];
        double L = norm(v);
        if (L > Lb)
        {
            Lb = L;
            eb = v;
        }
    }
    if (La < Lenmin || Lb < Lenmin)
        return -1.0;
    if (min(La, Lb) / max(La, Lb) < Changdubi)
        return -1.0;
    ea *= (float)(1.0 / La);
    eb *= (float)(1.0 / Lb);
    if (fabs(ea.x * eb.y - ea.y * eb.x) > SIN)
        return -1.0;
    Point2f ca = (a[0] + a[1] + a[2] + a[3]) * 0.25f;
    Point2f cb = (b[0] + b[1] + b[2] + b[3]) * 0.25f;
    Point2f cd = cb - ca;
    double cl = norm(cd);
    if (cl < 1e-6)
        return -1.0;
    double c1 = fabs((cd.x * ea.x + cd.y * ea.y) / cl);
    double c2 = fabs((cd.x * eb.x + cd.y * eb.y) / cl);
    if (max(c1, c2) > COS)
        return -1.0;
    return cl / max(La, Lb);
}
static void Miao(const Mat &img, Mat &out, Mat &mask)
{
    Mat hsv;                                                                // 用hsv分参
    cvtColor(img, hsv, COLOR_BGR2HSV);                                      // 转换图片类型，hsv 是最常用的抠图格式,因为可以把亮度饱和度和色彩分开
                                                                            // H 90-130是蓝色，调大往青蓝绿走，调小往紫红走    红色0-10橘红，170-179品红深红
    inRange(hsv, Scalar(Hmin, Smin, Vmin), Scalar(Hmax, Smax, Vmax), mask); // open蓝色

    // 开红色俩个都要开因为红色是0-10加上170-179
    // inRange(hsv,Scalar(Hmin,Smin,Vmin),Scalar(Hmax,Smax,Vmax),mask);
    // inRange(hsv,Scalar(0,Smin,Vmin),Scalar(10,Smax,Vmax),mask);

    // 闭合；先联接后消边
    morphologyEx(mask, mask, MORPH_CLOSE,
                 getStructuringElement(MORPH_RECT, Size(3, 3)));
    // 再加一次开运算，腐蚀掉孤立的小噪点
    morphologyEx(mask, mask, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(2, 2))); // size是处理的强度,不要开了，怕像素小了直接给灯条抹除了

    // 存矩形
    vector<vector<Point>> contours;                                   // 御用矩形存储形式
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // contours是各部分轮廓的存储数组，这个函数是把存储的点阵画到mask上
    out = img.clone();                                                // 克隆原图，用于画
    for (int i = 0; i < (int)contours.size(); i++)
    {
        RotatedRect box; // minAreaRect  传说中的旋转矩形
        box = minAreaRect(contours[i]);
        GYH(box, Pts[i]); // 取出四个点，排成 0左上 1左下 2右下 3右上
    } // 预处理想要的数据,如果不为单独拿出点的数据，结构体应该是可以省略的。

    int num = 0;
    for (int i = 0; i < (int)contours.size(); i++)
    {
        for (int j = i + 1; j < (int)contours.size(); j++)
        {
            double bili = BILI(Pts[i], Pts[j]);
            if (bili > BILI_MIN && bili < BILI_MAX)
            {
                R[num].A = i, R[num].B = j;
                num++;
            }
        }
    }

    if (!num)
        return;
   /* double maxx = 0;
    int maxp = 0; // 寻找合法的矩形对
      for (int i = 0; i < num; i++) // 找到最近矩形对
       {
           RotatedRect AA = minAreaRect(contours[R[i].A]);
           RotatedRect BB = minAreaRect(contours[R[i].B]);
           if (AA.size.height * AA.size.width + BB.size.height * BB.size.width > maxx)
           {
               maxx = AA.size.height * AA.size.width + BB.size.height * BB.size.width;
               maxp = i;
           }
       } */

    // 左右边中心点确定准星法
    /*
    if (Pts[R[maxp].A][0].x < Pts[R[maxp].B][0].x)
    {                                                         // A在左
        x1 = (Pts[R[maxp].A][3].x + Pts[R[maxp].B][0].x) / 2; // A的右上和B的坐上确定x1
        y1 = (Pts[R[maxp].A][3].y + Pts[R[maxp].B][1].y) / 2; // A的右上和b的左下确定y
    }
    else
    { // B在左
        x1 = (Pts[R[maxp].A][0].x + Pts[R[maxp].B][3].x) / 2;
        y1 = (Pts[R[maxp].A][1].y + Pts[R[maxp].B][3].y) / 2;
    }
    */
    // 俩矩形中心点确定准星法
    for (int i = 0; i < num; i++)
    {
        double x1 = 0, y1 = 0;
        RotatedRect boxA = minAreaRect(contours[R[i].A]);
        RotatedRect boxB = minAreaRect(contours[R[i].B]);
        x1 = (boxA.center.x + boxB.center.x) / 2;
        y1 = (boxA.center.y + boxB.center.y) / 2;
        cv::drawContours(out, contours, R[i].A, Scalar(0, 255, 0), 4); // 在这画，优化视觉体验喵
        cv::drawContours(out, contours, R[i].B, Scalar(0, 255, 0), 4);
        // 第二个是表示绘画的对象是哪个组，第三个是画 这个组里的第几个？ -1是全画。
        // rectangle不和旋转矩形匹配
        if (x1 && y1)
            circle(out, Point(x1, y1), 10, Scalar(0, 255, 0), -1);
        cv::waitKey(1); // 留帧设计
    }
}

int main(int argc, char **argv)
{
    VideoCapture cap;
    cap.open(CAMERA_INDEX); // open 摄像头
    bool op = cap.isOpened();
    if (!op)
    {
        cout << "没开摄像头！！！" << endl;
        return 0;
    }

    createHSVTrackbars(); // 创建滑动条

    cap.set(CAP_PROP_AUTO_EXPOSURE, 0);           // 关闭自动曝光 0关1开
    cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);   // 宽、长参数
    cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT); //
    cap.set(CAP_PROP_FPS, FRAME_FPS);             // fps设定
    cap.set(CAP_PROP_BUFFERSIZE, 1);              // 缓冲帧，竞赛设1？
    cap.set(CAP_PROP_BRIGHTNESS, BRIGHT);         // 亮度
    cap.set(CAP_PROP_EXPOSURE, BAO);              // 曝光

    while (true)
    {
        if (!cap.read(frame) || frame.empty())
        { // 读摄像头失败
            cout << "manba out" << endl;
            break;
        }
        Miao(frame, out, mask);
        namedWindow("fact", WINDOW_FREERATIO);
        namedWindow("test", WINDOW_FREERATIO);
        imshow("fact", out);
        imshow("test", mask);
        if (cv::waitKey(1) == 27)
            break;
        // esc退出
    }
    cap.release();       // 释放摄像头好习惯
    destroyAllWindows(); // 关闭窗口好习惯
    return 0;
}

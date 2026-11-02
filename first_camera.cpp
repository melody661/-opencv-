// ============================================================================
//  first_camera.cpp
//
//  依据“修正后的 first opcv.cpp”改写：检测流程与原文件逐行一致
//  （A[1000]、contourArea*4 的相对面积规则、k 计数、x1/=k），只是从 main 搬进
//  processFrame()，输入从 imread 换成摄像头取帧，每读一帧调用一次。
//
//  相对原文件只改了三件与环境相关的事：
//    · 去掉所有打印（cout / printf），不打任何提示；
//    · 打不开摄像头 / 读不到帧 / 没有可用轮廓时，直接 return 0，不再打印错误；
//    · imshow + waitKey 从“等一次按键”改成循环里每帧刷新。
//
//  编译：
//    g++ -std=c++17 -O2 first_camera.cpp ^
//        -I D:/opencv/build-mingw/install/include ^
//        -L D:/opencv/build-mingw/install/x64/mingw/lib ^
//        -lopencv_world490 -o first_camera.exe
//
//  运行：
//    first_camera.exe            实时处理，ESC / Q 退出
//    first_camera.exe --test 5   不弹窗，处理 5 帧后存 cam_out.png / cam_mask.png 退出
// ============================================================================

#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

// ============================ 可改的参数 ============================
static const int CAMERA_INDEX = 0;      // 摄像头编号
static const int FRAME_WIDTH = 640;
static const int FRAME_HEIGHT = 480;
static const int FRAME_FPS = 60;

static const char* WIN_RESULT = "mimicat";
static const char* WIN_MASK = "catmimi";
// ===================================================================

// 原文件的坐标结构体（原文件是 A[3]，修正后是 A[1000]）
struct zuobiao {
    double x;       // 左上角坐标
    double y;       // 左上角坐标
    double width;
    double hight;
} A[1000];

// ---------------------------------------------------------------------------
//  processFrame：原 first opcv.cpp 的检测流程，输入一帧，输出 out 和 mask
// ---------------------------------------------------------------------------
static void processFrame(const Mat& img, Mat& out, Mat& mask) {
    // ---------- 1. 用“高饱和度”把灯条芯部和光晕分开 ----------
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(0, 110, 230), Scalar(255, 255, 255), mask);

    // 闭运算：把被滤掉的灯条芯部补回来，变回实心长条
    morphologyEx(mask, mask, MORPH_CLOSE,
                 getStructuringElement(MORPH_RECT, Size(15, 15)));
    // 开运算：腐蚀掉孤立的小噪点
    morphologyEx(mask, mask, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));

    // ---------- 2. 找连通域 ----------
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    out = img.clone();

    // 摄像头对着别处时可能一个轮廓都没有：原代码会 x1/=0 得到 NaN 再画点
    if (contours.empty()) return;

    for (size_t i = 0; i < contours.size(); ++i) {
        Rect box = boundingRect(contours[i]);
        if (box.area() < 600) continue;                 // 小噪点不处理
        if (box.width < 10) continue;                   // 太细的条纹不处理
        if (box.height < 1.1 * box.width) continue;     // 灯条是竖长条

        // 判断红蓝
        Mat one = Mat::zeros(img.size(), CV_8U);
        drawContours(one, contours, (int)i, Scalar(255), FILLED);
        Scalar BGR = mean(img, one);                    // 求平均，用主体掩盖白光带来的误差
        double B = BGR[0], G = BGR[1], R = BGR[2];
        bool wtcolor = (R > B);
        Scalar color = wtcolor ? Scalar(0, 0, 255) : Scalar(255, 0, 0);   // BGR

        // 坐标
        A[i].x = box.x;
        A[i].y = box.y;
        A[i].hight = box.height;
        A[i].width = box.width;

        // 涂色 + 画框
        Mat overlay = out.clone();
        drawContours(overlay, contours, (int)i, color, FILLED);
        addWeighted(overlay, 0.70, out, 0.30, 0, out);  // 70% 覆盖，30% 原图
        rectangle(out, box, Scalar(0, 255, 0), -1);     // 画框
    }

    // ---------- 3. 平均坐标 ----------
    double x1 = 0, y1 = 0;
    int k = 0;
    double maxx = 0;
    for (int i = 0; i < (int)contours.size(); i++)
        maxx = max(maxx, contourArea(contours[i]));

    for (int i = 0; i < (int)contours.size(); i++) {
        if (contourArea(contours[i]) * 4 < maxx) continue;   // 小噪点不处理
        ++k;
        if (A[i].width > 10 && A[i].hight > 1.1 * A[i].width) {
            x1 += A[i].x + A[i].width / 2;
            y1 += A[i].y + A[i].hight / 2;
        }
    }

    // 没有任何一个轮廓通过面积规则时，原代码会除 0
    if (k == 0) return;

    x1 /= k;
    y1 /= k;
    circle(out, Point(x1, y1), 10, Scalar(0, 255, 0), -1);
}

// ---------------------------------------------------------------------------
//  打开摄像头：优先 DirectShow，失败再退回其它后端
// ---------------------------------------------------------------------------
static bool openCamera(VideoCapture& cap, int index) {
    for (int backend : {CAP_DSHOW, CAP_MSMF, CAP_ANY}) {
        if (cap.open(index, backend)) return true;
        cap.release();
    }
    return false;
}

int main(int argc, char** argv) {
    int testFrames = 0;
    for (int i = 1; i < argc; ++i)
        if (string(argv[i]) == "--test" && i + 1 < argc) testFrames = atoi(argv[++i]);

    VideoCapture cap;
    if (!openCamera(cap, CAMERA_INDEX)) return 0;

    cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cap.set(CAP_PROP_FPS, FRAME_FPS);
    cap.set(CAP_PROP_BUFFERSIZE, 1);

    Mat frame, out, mask;
    long long n = 0;

    while (true) {
        if (!cap.read(frame) || frame.empty()) return 0;

        // 每读到一帧，调用一次原检测流程
        processFrame(frame, out, mask);
        ++n;

        if (testFrames > 0) {
            if (n >= testFrames) {
                imwrite("cam_out.png", out);
                imwrite("cam_mask.png", mask);
                return 0;
            }
            continue;
        }

        imshow(WIN_RESULT, out);
        imshow(WIN_MASK, mask);
        int key = waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

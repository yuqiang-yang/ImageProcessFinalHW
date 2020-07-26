from PyQt5 import Qt
from PyQt5 import QtCore,QtWidgets,QtGui
import sys
import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QGraphicsRectItem, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QSize
import cv2
import numpy as np
from matplotlib import pyplot as plt
from  PyQt5.QtMultimedia import  QCamera,QCameraImageCapture,QCameraViewfinderSettings
from PIL import  Image
import mycamera
import math
import dlib

class MainWindow():
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        self.raw_image = None
        self.ui = mycamera.Ui_MainWindow()
        self.ui.setupUi(MainWindow)
        self.captured_image = None                  #摄像头捕获的图片
        self.timer = QtCore.QTimer()                #创建定时器，用于捕获视频帧
        self.cap = cv2.VideoCapture()
        self.predictor_path = 'shape_predictor_68_face_landmarks.dat'       #人脸检测数据
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

        self.action_connect()                       #信号槽绑定

        MainWindow.show()
        sys.exit(app.exec_())

    #信号槽绑定
    def action_connect(self):
        self.ui.action_3.triggered.connect(self.open_file)          #打开图片
        self.ui.action_4.triggered.connect(self.save_file)          #保存图片
        self.ui.pushButton.clicked.connect(self.open_camera)        #打开相机
        self.ui.pushButton_2.clicked.connect(self.capture_picture)  #拍照
        self.timer.timeout.connect(self.show_camera)                #显示摄像头
        self.ui.horizontalSlider.valueChanged.connect(self.slider_change)         #亮度
        self.ui.horizontalSlider_2.valueChanged.connect(self.slider_change)       #对比度
        self.ui.horizontalSlider_3.valueChanged.connect(self.slider_change)       #饱和度
        self.ui.horizontalSlider_4.valueChanged.connect(self.slider_change)       #曝光度
        self.ui.horizontalSlider_5.valueChanged.connect(self.slider_change)       #模糊
        self.ui.horizontalSlider_10.valueChanged.connect(self.slider_change)
        self.ui.horizontalSlider_11.valueChanged.connect(self.slider_change)
        self.ui.horizontalSlider_12.valueChanged.connect(self.slider_change)
        self.ui.radioButton.clicked.connect(self.slider_change)
        self.ui.radioButton_2.clicked.connect(self.slider_change)
        self.ui.radioButton_3.clicked.connect(self.slider_change)
        self.ui.radioButton_4.clicked.connect(self.slider_change)
        self.ui.radioButton_5.clicked.connect(self.slider_change)
        self.ui.radioButton_6.clicked.connect(self.slider_change)
        self.ui.radioButton_7.clicked.connect(self.slider_change)
        self.ui.radioButton_8.clicked.connect(self.slider_change)
        self.ui.radioButton_9.clicked.connect(self.slider_change)
        self.ui.radioButton_10.clicked.connect(self.slider_change)
        self.ui.radioButton_11.clicked.connect(self.slider_change)
        self.ui.radioButton_12.clicked.connect(self.slider_change)
        self.ui.radioButton_13.clicked.connect(self.slider_change)
        self.ui.radioButton_14.clicked.connect(self.slider_change)

    #打开文件
    def open_file(self):
        filename = QFileDialog.getOpenFileName(None,'选择图片文件','./',("Images (*.png *.xpm *.jpg *.jped *.bmp)"))
        if filename[0]:         #假如选中了文件
            img = cv2.imdecode(np.fromfile(filename[0],dtype=np.uint8),-1)      #读取RBG色彩空间的图像
            self.filename = filename[0]
            self.bgr_img = cv2.imread(filename[0],cv2.IMREAD_COLOR)
            self.raw_image = img
            self.current_img = img
            self.show_image()

    #保存文件
    def save_file(self):
        filename = QFileDialog.getSaveFileName(None,'保存图片','./',("Images (*.png *.xpm *.jpg *.jped *.bmp)"))
        if filename[0]:
            cv2.imwrite('process_img.png',self.current_img)         #这里有点bug，直接用filename[0]无法正常保存

    #显示图片
    def show_image(self):
        img = cv2.cvtColor(self.current_img,cv2.COLOR_RGB2BGR)       #把RGB图片转为BGR格式，用于opencv读取
        width,height,demension = img.shape                          #获取图片的尺寸
        img_ratio = width/height                                    #图片的宽高比例
        window_ratio = self.ui.graphicsView.width()/self.ui.graphicsView.height()   #显示窗口的宽高比例
        if img_ratio > window_ratio:
            width = int(self.ui.graphicsView.width())
            height = int(self.ui.graphicsView.width() / img_ratio)  #高度缩放
        else:
            width = int(self.ui.graphicsView.height() * img_ratio)  #宽度拉伸
            height = int(self.ui.graphicsView.height())
        img_transfer = cv2.resize(img,(height-5,width - 5),interpolation=cv2.INTER_AREA)  #对原始图片进行缩放处理，用于在窗口中显示
        h,w,c = img_transfer.shape                                  #得到转换后图片的尺寸
        q_img = QImage(img_transfer.data,w,h,w*3,QImage.Format_RGB888)      #创建QImage对象
        self.window = QGraphicsScene()
        pix = QPixmap(q_img)
        self.window.addPixmap(pix)
        self.ui.graphicsView.setScene(self.window)                   #显示图片
    #打开摄像头
    def open_camera(self):
        if self.timer.isActive() == False:
            flag = self.cap.open(0)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self,'warning',"请检查相机于电脑是否连接正确",buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer.start(20)
                self.ui.pushButton.setText('关摄像头')
        else:
            self.timer.stop()
            self.cap.release()
            self.ui.graphicsView.clear()

    #显示摄像头
    def show_camera(self):
        flag, self.image = self.cap.read()                      #获取摄像头画面
        show = cv2.resize(self.image, (700, 500))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)            #转为RGB才是现实的色彩
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.window = QGraphicsScene()
        pix = QPixmap(showImage)
        self.window.addPixmap(pix)
        self.ui.graphicsView.setScene(self.window)                   #显示图片

    #拍摄照片
    def capture_picture(self):
        self.current_img = self.image
        self.raw_image = self.image
        self.timer.stop()                   #拍完照后关闭定时器
        self.cap.release()                  #关闭摄像头
        self.show_image()                   #显示拍摄图片
        cv2.imwrite('camera.png',self.current_img)
        self.filename = 'camera.png'

    #亮度
    def change_brightness(self):
        value = self.ui.horizontalSlider.value()
        hls_img = cv2.cvtColor(self.current_img,cv2.COLOR_RGB2HLS)
        if value > 2:
            hls_img[:, :, 1] = np.log(hls_img[:, :, 1] /255* (value - 1)+1) / np.log(value + 1) * 255
        if value < 0:
            hls_img[:, :, 1] = np.uint8(hls_img[:, :, 1] / np.log(- value + np.e))
        self.current_img = cv2.cvtColor(hls_img,cv2.COLOR_HLS2RGB)


    #对比度
    def change_contrast(self):
        value = self.ui.horizontalSlider_2.value()
        img = self.current_img
        self.current_img = cv2.normalize(img,dst=None,alpha=value*10,beta=10,norm_type=cv2.NORM_MINMAX)

    #饱和度
    def change_saturation(self):
        value = self.ui.horizontalSlider_3.value()
        hls_img = cv2.cvtColor(self.current_img, cv2.COLOR_RGB2HLS)
        if value > 1:
            hls_img[:, :, 2] = np.log(hls_img[:, :, 2] /255* (value - 1)+1) / np.log(value + 1) * 255
        if value < 0:
            hls_img[:, :, 2] = np.uint8(hls_img[:, :, 2] / np.log(- value + np.e))
        self.current_img = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB)

    #色相
    def change_hue(self):
        value = self.ui.horizontalSlider_4.value()
        hls_img = cv2.cvtColor(self.current_img,cv2.COLOR_RGB2HLS)
        if value > 2:
            hls_img[:, :, 0] = np.log(hls_img[:, :, 0] /255* (value - 1)+1) / np.log(value + 1) * 255
        if value < 0:
            hls_img[:, :, 0] = np.uint8(hls_img[:, :, 0] / np.log(- value + np.e))
        self.current_img = cv2.cvtColor(hls_img,cv2.COLOR_HLS2RGB)
    #模糊
    def blur(self):
        value = self.ui.horizontalSlider_5.value()
        self.current_img = cv2.GaussianBlur(self.current_img,(7,7),sigmaX=value/8.0,sigmaY=value/8.0)

    #锐化
    def remove_sharpen(self):
        pass
    #拉普拉斯算子锐化
    def laplacian_sharpen(self):
        kenerl1 = np.array(([0,-1,0],[-1,5,-1],[0,-1,0]),dtype="float32")
        img = self.current_img
        self.current_img = cv2.filter2D(img,ddepth=cv2.CV_8U,kernel=kenerl1)

    #UMS方法锐化
    def UMS_sharpen(self):
        img = cv2.GaussianBlur(self.current_img,(5,5),sigmaX=50,sigmaY=50)
        img2 = self.current_img
        self.current_img = cv2.addWeighted(img2,1.5,img,-0.5,0)
    #浮雕效果
    def remove_relief(self):
        pass
    #彩色浮雕
    def colored_relief(self):
        kenerl1 = np.array(([-1,0,0],[0,1,0],[0,0,0]),dtype="float32")
        img = self.current_img
        self.current_img = cv2.filter2D(img,ddepth=cv2.CV_8U,kernel=kenerl1)+128
    #灰色浮雕
    def gray_relief(self):
        kenerl1 = np.array(([-1,0,0],[0,1,0],[0,0,0]),dtype="float32")
        img = self.current_img
        img2 = cv2.filter2D(img,ddepth=cv2.CV_8U,kernel=kenerl1)+128
        self.current_img = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

    #怀旧滤镜
    def old_filter(self):
        img = np.asarray(Image.open(self.filename).convert('RGB'))
        trans = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]).transpose()
        img = np.dot(img,trans).clip(max=255)
        self.current_img = np.array(img).astype('uint8')
    #流年滤镜
    def years_filter(self):
        rows,cols,c = self.current_img.shape
        img = np.zeros((rows,cols,3),dtype='uint8')
        for i in range(rows):
            for j in range(cols):
                B = math.sqrt(self.current_img[i,j][0])*12
                G = self.current_img[i,j][1]
                R = self.current_img[i,j][2]
                if B >255:
                    B = 255
                img[i,j] = np.uint8((B,G,R))
        self.current_img = img
    #连环画滤镜
    def comic_filter(self):
        rows,cols,c = self.current_img.shape
        img = np.zeros((rows,cols,3),dtype='uint8')
        for i in range(rows):
            for j in range(cols):
                B = int(self.current_img[i,j][0])
                G = int(self.current_img[i,j][1])
                R = int(self.current_img[i,j][2])
                R1 = abs(G-B+G+R)*R/256
                G1 = abs(B-G+B+R)*R/256
                B1 = abs(B-G+B+R)*G/256
                if R1 > 255:
                    R1 = 255
                if G1 > 255:
                    G1 = 255
                if B1 > 255:
                    B1 = 255
                img[i,j] = np.uint8((B1,G1,R1))
        self.current_img = img
        pass
    #冰冻滤镜
    def frozen_filter(self):
        rows,cols,c = self.current_img.shape
        img = np.zeros((rows,cols,3),dtype='uint8')
        for i in range(rows):
            for j in range(cols):
                B = int(self.current_img[i,j][0])
                G = int(self.current_img[i,j][1])
                R = int(self.current_img[i,j][2])
                B1 = abs((B-R-G)/2.0*3)
                G1 = abs((G-B-R)/2.0*3)
                R1 = abs((R-B-G)/2.0*3)
                if B1 >255:
                    B1 = 255
                if G1 > 255:
                    G1 = 255
                if R1 >255:
                    R1 = 255
                img[i,j] = np.uint8((B1,G1,R1))
        self.current_img = img
    #去色滤镜
    def remove_color_filter(self):
        rows,cols,c = self.current_img.shape
        img = np.zeros((rows,cols,3),dtype='uint8')
        for i in range(rows):
            for j in range(cols):
                B = int(self.current_img[i,j][0])
                G = int(self.current_img[i,j][1])
                R = int(self.current_img[i,j][2])
                pix = (max(B,G,R) + min(B,G,R))/2.0
                img[i,j] = np.uint8((pix,pix,pix))
        self.current_img = img

    #皮肤检测
    def skin_detect(self):
        img = self.current_img
        img_ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        (y,cr,cb) = cv2.split(img_ycrcb)
        cr1 = cv2.GaussianBlur(cr,(5,5),0)
        _,skin = cv2.threshold(cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.current_img = skin
    #人脸检测
    def face_detect(self):
        img = self.raw_image
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            self.current_img = cv2.rectangle(self.current_img.copy(), (x, y), (x + w, y + h), (255, 0, 0), 1)
        pass
    #人脸特征点检测
    def landmark_detect(self,img):
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        land_marks = []

        rects = self.detector(img_gray,0)

        for i in range(len(rects)):
            land_marks_node = np.matrix([[p.x, p.y] for p in self.predictor(img_gray, rects[i]).parts()])
            # for idx,point in enumerate(land_marks_node):
            #     # 68点坐标
            #     pos = (point[0,0],point[0,1])
            #     print(idx,pos)
            #     # 利用cv2.circle给每个特征点画一个圈，共68个
            #     cv2.circle(img_src, pos, 5, color=(0, 255, 0))
            #     # 利用cv2.putText输出1-68
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     cv2.putText(img_src, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            land_marks.append(land_marks_node)
            return land_marks
    #局部变换
    def localTranslationWarp(self,srcImg, startX, startY, endX, endY, radius):

        ddradius = float(radius * radius)
        copyImg = np.zeros(srcImg.shape, np.uint8)
        copyImg = srcImg.copy()

        # 计算公式中的|m-c|^2
        ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
        H, W, C = srcImg.shape
        for i in range(W):
            for j in range(H):
                # 计算该点是否在形变圆的范围之内
                # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中
                if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                    continue

                distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)

                if (distance < ddradius):
                    # 计算出（i,j）坐标的原坐标
                    # 计算公式中右边平方号里的部分
                    ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                    ratio = ratio * ratio

                    # 映射原位置
                    UX = i - ratio * (endX - startX)
                    UY = j - ratio * (endY - startY)

                    # 根据双线性插值法得到UX，UY的值
                    value = self.BilinearInsert(srcImg, UX, UY)
                    # 改变当前 i ，j的值
                    copyImg[j, i] = value

        return copyImg
    #双线性插值法
    def BilinearInsert(self,src, ux, uy):
        w, h, c = src.shape
        if c == 3:
            x1 = int(ux)
            x2 = x1 + 1
            y1 = int(uy)
            y2 = y1 + 1

            part1 = src[y1, x1].astype(np.float) * (float(x2) - ux) * (float(y2) - uy)
            part2 = src[y1, x2].astype(np.float) * (ux - float(x1)) * (float(y2) - uy)
            part3 = src[y2, x1].astype(np.float) * (float(x2) - ux) * (uy - float(y1))
            part4 = src[y2, x2].astype(np.float) * (ux - float(x1)) * (uy - float(y1))

            insertValue = part1 + part2 + part3 + part4

            return insertValue.astype(np.int8)
    #瘦脸总调用函数
    def face_thin_auto(self,src):

        landmarks = self.landmark_detect(src)

        # 如果未检测到人脸关键点，就不进行瘦脸
        if len(landmarks) == 0:
            return

        for landmarks_node in landmarks:
            left_landmark = landmarks_node[3]
            left_landmark_down = landmarks_node[5]

            right_landmark = landmarks_node[13]
            right_landmark_down = landmarks_node[15]

            endPt = landmarks_node[30]

            # 计算第4个点到第6个点的距离作为瘦脸距离
            r_left = math.sqrt(
                (left_landmark[0, 0] - left_landmark_down[0, 0]) * (left_landmark[0, 0] - left_landmark_down[0, 0]) +
                (left_landmark[0, 1] - left_landmark_down[0, 1]) * (left_landmark[0, 1] - left_landmark_down[0, 1]))

            # 计算第14个点到第16个点的距离作为瘦脸距离
            r_right = math.sqrt((right_landmark[0, 0] - right_landmark_down[0, 0]) * (
                        right_landmark[0, 0] - right_landmark_down[0, 0]) +
                                (right_landmark[0, 1] - right_landmark_down[0, 1]) * (
                                            right_landmark[0, 1] - right_landmark_down[0, 1]))

            # 瘦左边脸
            thin_image = self.localTranslationWarp(src, left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1],
                                              r_left)
            # 瘦右边脸
            thin_image = self.localTranslationWarp(thin_image, right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0],
                                              endPt[0, 1], r_right)
            return thin_image
     #水印（未用）
    def change_big_eye(self):
        rows,cols,c = self.current_img.shape
        #cv2.putText(self.current_img,'yuqiang',(rows-50,cols-50),4,2,(0,0,0))
        pass

    #磨皮祛痘
    def dermabrasion(self):
        value = self.ui.horizontalSlider_12.value()
        img = self.current_img
        if img.shape[2] == 4:
            img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)

        img = cv2.GaussianBlur(img,(9,9),0,0)
        img3 = cv2.bilateralFilter(img,value,value*2,value/2)
        img4 = cv2.GaussianBlur(img3,(0,0),9)
        self.current_img = cv2.addWeighted(img3,1.5,img4,-0.5,0)
        pass
    #瘦脸
    def thin_face(self):
        img = self.current_img
        self.current_img = self.face_thin_auto(img)

    #水印（效果不好）
    def watermark(self, alpha=1):
        img = self.current_img
        h, w = self.current_img.shape[0], self.current_img.shape[1]
        mask = cv2.imread('mask.png', -1)

        if w > h:
            rate = int(w * 0.1) / mask.shape[1]
        else:
            rate = int(h * 0.1) / mask.shape[0]

        mask = cv2.resize(mask, None, fx=rate, fy=rate)
        mask_h, mask_w = mask.shape[0], mask.shape[1]
        mask_channels = cv2.split(mask)
        dst_channels = cv2.split(img)
        b, g, r, a = cv2.split(mask)

        # 计算mask在图片的坐标
        ul_points = (int(h * 0.9), int(int(w / 2) - mask_w / 2))
        dr_points = (int(h * 0.9) + mask_h, int(int(w / 2) + mask_w / 2))
        for i in range(3):
            dst_channels[i][ul_points[0]: dr_points[0], ul_points[1]: dr_points[1]] = dst_channels[i][
                                                                                      ul_points[0]: dr_points[0],
                                                                                      ul_points[1]: dr_points[1]] * (
                                                                                                  255.0 - a * alpha) / 255
            dst_channels[i][ul_points[0]: dr_points[0], ul_points[1]: dr_points[1]] += np.array(
                mask_channels[i] * (a * alpha / 255), dtype=np.uint8)
        dst_img = cv2.merge(dst_channels)
        # cv2.imwrite(r'd:\1_1.jpg', dst_img)
        self.current_img = dst_img
# 响应滑动条的变化
    def slider_change(self):
        if self.raw_image is None:
            return 0
        self.current_img = self.raw_image
        # 亮度
        if self.ui.horizontalSlider.value() != 0:
            self.change_brightness()
        # 对比度
        if self.ui.horizontalSlider_2.value() != 0:
            self.change_contrast()
        # 饱和度
        if self.ui.horizontalSlider_3.value() != 0:
            self.change_saturation()
        # 色相
        if self.ui.horizontalSlider_4.value() != 0:
            self.change_hue()
        # 模糊
        if self.ui.horizontalSlider_5.value() != 0:
            self.blur()
        #锐化
        if self.ui.radioButton.isChecked() == True:
            pass
        #拉普拉斯算子锐化
        if self.ui.radioButton_2.isChecked() == True:
            self.laplacian_sharpen()
        #UMS方法锐化
        if self.ui.radioButton_3.isChecked() == True:
            self.UMS_sharpen()
        #浮雕
        if self.ui.radioButton_4.isChecked() == True:
            pass
        #彩色浮雕
        if self.ui.radioButton_5.isChecked() == True:
            self.colored_relief()
        #灰色浮雕
        if self.ui.radioButton_6.isChecked() == True:
            self.gray_relief()
        #滤镜
        if self.ui.radioButton_7.isChecked() == True:
            pass
        #流年滤镜
        if self.ui.radioButton_8.isChecked() == True:
            self.years_filter()
        #连环画滤镜
        if self.ui.radioButton_9.isChecked() == True:
            self.comic_filter()
        #怀旧滤镜
        if self.ui.radioButton_10.isChecked() == True:
            self.old_filter()
        #冰冻滤镜
        if self.ui.radioButton_11.isChecked() == True:
            self.frozen_filter()
        #去色滤镜
        if self.ui.radioButton_12.isChecked() == True:
            self.remove_color_filter()
        #皮肤检测
        if self.ui.radioButton_13.isChecked() == True:
            self.skin_detect()
        #人脸检测
        if self.ui.radioButton_14.isChecked() == True:
            self.face_detect()
        #瘦脸
        if self.ui.horizontalSlider_10.value() != 0:
            self.thin_face()
        #水印（未甩）
        if self.ui.horizontalSlider_11.value() != 0:
            pass
            #self.watermark()
        #磨皮祛痘
        if self.ui.horizontalSlider_12.value() != 0:
            self.dermabrasion()
        self.show_image()


if __name__ == "__main__":
    MainWindow()
    
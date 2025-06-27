import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import winsound

class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("교통 약자 표지판 인식")
        self.setGeometry(200, 200, 700, 200)

        signButton = QPushButton('표지판 불러오기', self)
        roadButton = QPushButton('도로 영상 불러오기', self)
        recognitionButton = QPushButton('인식하기', self)
        quitButton = QPushButton('종료', self)

        self.label = QLabel('결과 표시 영역', self)

        signButton.setGeometry(10, 10, 150, 30)
        roadButton.setGeometry(170, 10, 150, 30)
        recognitionButton.setGeometry(330, 10, 150, 30)
        quitButton.setGeometry(490, 10, 100, 30)
        self.label.setGeometry(10, 50, 680, 130)

        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.signFiles = [
            ['child.png', '어린이 보호구역'],
            ['elder.png', '노인 보호구역'],
            ['disabled.png', '장애인 보호구역']
        ]
        self.signImgs = []
        self.roadImg = None

    def signFunction(self):
        self.label.clear()
        self.label.setText("표지판 이미지를 불러왔습니다.")

        for fname, _ in self.signFiles:
            img = cv.imread(fname)
            if img is not None:
                self.signImgs.append(img)
                cv.imshow(fname, img)

    def roadFunction(self):
        if not self.signImgs:
            self.label.setText('먼저 표지판을 불러오세요.')
            return

        fname = QFileDialog.getOpenFileName(self, '도로 영상 열기', './')
        if fname[0]:
            self.roadImg = cv.imread(fname[0])
            if self.roadImg is None:
                sys.exit('로드된 도로 이미지가 없습니다.')
            cv.imshow('Road Scene', self.roadImg)

    def recognitionFunction(self):
        if self.roadImg is None:
            self.label.setText('도로 영상을 먼저 불러오세요.')
            return

        sift = cv.SIFT_create()
        key_desc_list = []

        for img in self.signImgs:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            key_desc_list.append((kp, des))

        grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
        road_kp, road_des = sift.detectAndCompute(grayRoad, None)

        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        good_matches_all = []

        for sign_kp, sign_des in key_desc_list:
            knn_match = matcher.knnMatch(sign_des, road_des, 2)

            T = 0.7
            good_match = []

            for nearest1, nearest2 in knn_match:
                if nearest1.distance / nearest2.distance < T:
                    good_match.append(nearest1)
            good_matches_all.append(good_match)

        best_idx = good_matches_all.index(max(good_matches_all, key=len))

        if len(good_matches_all[best_idx]) < 4:
            self.label.setText('특징점이 충분하지 않습니다.')
        else:
            sign_kp = key_desc_list[best_idx][0]
            good_match = good_matches_all[best_idx]

            points1 = np.float32([sign_kp[m.queryIdx].pt for m in good_match])
            points2 = np.float32([road_kp[m.trainIdx].pt for m in good_match])

            H, _ = cv.findHomography(points1, points2, cv.RANSAC)

            h1, w1 = self.signImgs[best_idx].shape[:2]
            h2, w2 = self.roadImg.shape[:2]

            box1 = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
            box2 = cv.perspectiveTransform(box1, H)

            self.roadImg = cv.polylines(self.roadImg, [np.int32(box2)], True, (0, 255, 0), 4)

            img_match = np.empty((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            cv.drawMatches(self.signImgs[best_idx], sign_kp, self.roadImg, road_kp, good_match, img_match,
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv.imshow('Matches and Homography', img_match)

            self.label.setText(self.signFiles[best_idx][1] + ' 표지판이 인식되었습니다!')
            winsound.Beep(3000, 500)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
win = TrafficWeak()
win.show()
app.exec_()

from PyQt5.QtWidgets import *
import cv2 as cv
import numpy as np
import winsound
import sys

class Panorama(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('파노라마 이미지')
        self.setGeometry(200, 200, 700, 200)

        loadButton = QPushButton('이미지 불러오기', self)
        self.stitchButton = QPushButton('파노라마 생성', self)
        self.saveButton = QPushButton('저장', self)
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        loadButton.setGeometry(10, 25, 150, 30)
        self.stitchButton.setGeometry(170, 25, 150, 30)
        self.saveButton.setGeometry(330, 25, 150, 30)
        quitButton.setGeometry(490, 25, 150, 30)
        self.label.setGeometry(10, 100, 600, 30)

        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)

        loadButton.clicked.connect(self.loadImages)
        self.stitchButton.clicked.connect(self.stitchFunction)
        self.saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

    def loadImages(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, '이미지 파일 선택', './', "Images (*.png *.jpg *.jpeg)")
        if fnames:
            self.imgs = []
            for fname in fnames:
                img = cv.imread(fname)
                if img is not None:
                    self.imgs.append(img)
            if len(self.imgs) >= 2:
                self.label.setText(f"{len(self.imgs)}장의 이미지를 불러왔습니다.")
                self.stitchButton.setEnabled(True)
            else:
                self.label.setText("이미지는 최소 2장 이상 선택해야 합니다.")
        else:
            self.label.setText("이미지 선택이 취소되었습니다.")

    def stitchFunction(self):
        stitcher = cv.Stitcher_create()
        status, self.img_stitched = stitcher.stitch(self.imgs)
        if status == cv.Stitcher_OK:
            cv.imshow("Image Stitched Panorama", self.img_stitched)
            self.label.setText("파노라마 생성이 완료되었습니다!")
            self.saveButton.setEnabled(True)
        else:
            winsound.Beep(3000, 500)
            self.label.setText("파노라마 생성에 실패했습니다. 다시 시도해주세요.")

    def saveFunction(self):
        fname, _ = QFileDialog.getSaveFileName(self, '파일 저장', './', "Images (*.png *.jpg *.jpeg)")
        if fname:
            cv.imwrite(fname, self.img_stitched)
            self.label.setText("이미지가 저장되었습니다.")

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Panorama()
    win.show()
    app.exec_()

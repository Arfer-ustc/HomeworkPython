from graphics import *

class Switch():
    def __init__(self):
        self.inputfile="work3.gif"
        self.img=Image(Point(200,200),self.inputfile)
        self.width=self.img.getWidth()
        self.height=self.img.getHeight()

        #弹出一个新的窗口
        self.windowImg=GraphWin('Color Image', self.width*2, self.height*2)
        self.img.undraw()
        self.img.draw(self.windowImg)


    def convert(self):
        image=self.img
        self.windowImg.getMouse()

        for row in range(image.getHeight()):
            for column in range(image.getWidth()):
                r, g, b = image.getPixel(row, column)
                brightness = int(round(0.299 * r + 0.587 * g + 0.114 * b))
                image.setPixel(row, column, color_rgb(brightness, brightness, brightness))


        text = Text(Point(200, 50), "转换成功，单击鼠标进行保存！")
        text.draw(self.windowImg)
        self.windowImg.getMouse()


    def savaGray(self):
        newwindow=GraphWin("将图片存储到：",400,400)
        Text(Point(200,150),'输入文件名称：').draw(newwindow)
        Text(Point(200,250), '退出').draw(newwindow)
        inputText=Entry(Point(200,200),10)
        Text(Point(250,200),".gif").draw(newwindow)
        inputText.setText("gray")
        inputText.draw(newwindow)
        newwindow.getMouse()
        outputFile=inputText.getText()
        self.img.save(outputFile+'.gif')




if __name__ == '__main__':
    #初始化输出的原始图片，获得图片对象并显示原图片
    switch=Switch()
    #转化为灰度图
    switch.convert()
    #将灰度图保存
    switch.savaGray()


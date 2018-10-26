# 导入其他的类. 根据功能要求, 将不同的组件划分为不同的类, 并放入不同的文件, 简化主程序
from graphics import *
from button import Button
from homework.forth.second.shot_tracker import ShotTracker
from homework.forth.second.input_dialog import InputDialog


# 主程序
def main():
    # 新建主体窗口, 大小为 640x480, 并关闭自动界面刷新
    win = GraphWin('Animation', 640, 480, autoflush=False)

    # 坐标重映射
    # 将 X 轴的 [0, 640] 映射为 [-10, 210]
    # 将 Y 轴的 [0, 480] 映射为 [-10, 155]
    win.setCoords(-10, -10, 210, 155)

    # 画出数轴基线
    Line(Point(-10, 0), Point(210, 0)).draw(win)
    for x in range(0, 210, 50):
        Text(Point(x, -5), str(x)).draw(win)
        Line(Point(x, 0), Point(x, 2)).draw(win)

    angle,vel,height=45.0,40.0,2.0
    while True:
        # 获取用户输入, 设置发射角度、初始速度、初始高度
        inputWin = InputDialog(90, 30, 0)

        # 获取下一步选项
        choice = inputWin.interact()

        # 关闭用户输入窗口
        inputWin.close()
        if choice =='Quit':
            break

        # choice 为 'Fire' 则进入模拟发射过程
        # choice 为 'Quit' 则退出程序

        # 设置发射角度、初始速度、初始高度
        angle, vel, height = inputWin.getValues()

        # shot 就是用于模拟炮弹发射过程的对象
        shot = ShotTracker(win, angle, vel, height)

        # 当 shot 的 Y > 0 (位于地平面以上), 且 -10<X<210 (炮弹在显示窗口内)
        while 0 <= shot.getY() and shot.getX() > -10 and shot.getX() <= 210:
            # 以 1/30 s 的时间间隔不断更新 shot 坐标
            shot.update(1 / 50)

            # 窗口刷新频率设为 30fps
            update(50)
        maxHeight = "抛物线运动的最大高度: %.2f 米" % shot.proj.getMaxHeight()
        newWin = GraphWin('Max Height', 240, 180)
        Text(Point(120, 90), maxHeight).draw(newWin)
    win.close()



if __name__ == '__main__':
    main()
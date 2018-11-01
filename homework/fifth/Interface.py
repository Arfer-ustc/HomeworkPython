from graphics import GraphWin, Text, Point

from homework.fifth.dieview import DieView
from homework.fifth.button import Button


class Graphicsinterface:
    def __init__(self):
        self.win = GraphWin("Dice Poker", 600, 400)
        self.win.setBackground("green3")
        banner = Text(Point(300, 30), "Python Poker Parlor")
        banner.setSize(24)
        banner.setFill("yellow2")
        banner.setStyle('bold')
        banner.draw(self.win)

        self.msg = Text(Point(300, 380), "Welcome to the Dice Table")
        self.msg.setSize(18)
        self.msg.draw(self.win)
        self.createDice(Point(300, 100), 75)
        self.buttons = []
        self.addDiceButtons(Point(300, 170), 75, 30)
        b = Button(self.win, Point(300, 230), 400, 40, "Roll Dice")
        self.buttons.append(b)
        b = Button(self.win, Point(300, 280), 150, 40, "Score")
        self.buttons.append(b)
        b = Button(self.win, Point(570, 375), 40, 30, "Quit")
        self.buttons.append(b)
        self.money = Text(Point(300, 325), "$100")
        self.money.setSize(18)
        self.money.draw(self.win)

    def createDice(self, center, size):
        center.move(- 3 * size,0)
        self.dice = []
        for i in range(5):
            view = DieView(self.win, center, size)
            self.dice.append(view)
            center.move(1.5 * size,0)

    def addDiceButtons(self, center, width, height):
        center.move(-3 * width, 0)
        for i in range(1, 6):
            label = "Die {0}".format(i)
            b = Button(self.win, center, width, height, label)
            self.buttons.append(b)
            center.move(1.5 * width, 0)

    def setMoney(self, amt):
        self.money.setText("${0}".format(amt))

    def showResult(self, msg, score):
        if score > 0:
            text="{0}! You win ${1}".format(msg, score)
        else:
            text="You rolled {0}".format(msg)
        self.msg.setText(text)

    def setDice(self, values):
        for i in range(5):
            self.dice[i].setValue(values[i])
    def wantToPlay(self):
        ans = self.choose(["Roll Dice", "Quit"])
        self.msg.setText("")
        return ans == "Roll Dice"

    def choose(self, choices):
        buttons = self.buttons
        # activate choice buttons, deactivate others
        for b in buttons:
            if b.getLabel() in choices:
                b.activate()
            else:
                b.deactivate()
        # get mouse clicks until an active button is clicked
        while True:
            p=self.win.getMouse()
            for b in buttons:
                if b.clicked(p):
                    return b.getLabel()
    def chooseDice(self):
        choices=[]
        while True:
            b=self.choose(["Die 1","Die 2","Die 3","Die 4","Die 5",
                           "Roll Dice","Score"])
            if b[0]=="D":
                i=int(b[4])-1
                if i in choices:
                    choices.remove(i)
                    self.dice[i].setColor("black")
                else:
                    choices.append(i)
                    self.dice[i].setColor("gray")
            else:
                for d in self.dice:
                    d.setColor("black")
                if b=="Score":
                    return []
                elif choices!=[]:
                    return choices
    def close(self):
        self.win.close()

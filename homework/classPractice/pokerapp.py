from graphics import GraphWin, Text, Point

from homework.classPractice.Dice import Dice


class PokerApp:
    def __init__(self,interface):
        self.dice = Dice()
        self.money = 100
        self.interface = interface

    def run(self):
        while self.money >= 10 and self.interface.wantToPlay()!="Quit":
            code=self.interface.wantToPlay()

            if code=="rule":
                self.showRelu()
            else:
                self.playRound()
        # while self.money >= 10 and self.interface.wantToKnow():
        #     self.showRelu()
        self.interface.close()

    def playRound(self):
        self.money = self.money - 10
        self.interface.setMoney(self.money)
        self.doRolls()
        result, score = self.dice.score()
        self.interface.showResult(result, score)
        self.money = self.money + score
        self.interface.setMax(self.money)
        self.interface.setMoney(self.money)
    def showRelu(self):
        # self.interface.showRule()
        maxHeight = "游戏规则如下: "+"\n"+ \
                    "1_The player starts with $100" + "\n" + \
                    "2_Each round costs $10,subtracted from the player's money at the start" + "\n" + \
                    "3_All five dice are rolled randomly" + "\n" + \
                    "4_The player gets two chances to enhance the hand by rerolling some or all of the dice" + "\n" + \
                    "5_At the end of the hand, the player's money is updated."
        newWin = GraphWin('Max Height', 440, 280)
        Text(Point(220,120), maxHeight).draw(newWin)

    def doRolls(self):
        self.dice.rollAll()

        roll = 1
        self.interface.setDice(self.dice.values())
        toRoll = self.interface.chooseDice()
        while roll < 3 and toRoll != []:
            self.dice.roll(toRoll )
            roll = roll + 1
            self.interface.setDice(self.dice.values())
            if roll < 3:
                toRoll = self.interface.chooseDice()
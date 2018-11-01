from homework.classPractice.Interface import Graphicsinterface
from homework.classPractice.pokerapp import PokerApp
from homework.classPractice.textpoker import TextInterface
# inter = TextInterface()

if __name__ == '__main__':
    inter = Graphicsinterface()
    app = PokerApp(inter)
    app.run()
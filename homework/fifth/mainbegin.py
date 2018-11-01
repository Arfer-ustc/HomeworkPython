from homework.fifth.Interface import Graphicsinterface
from homework.fifth.pokerapp import PokerApp
from homework.fifth.textpoker import TextInterface
# inter = TextInterface()
inter=Graphicsinterface()
app = PokerApp(inter)
app.run()
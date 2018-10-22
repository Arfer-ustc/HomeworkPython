from PIL import Image
import matplotlib.pyplot as plt

def showGray(inputfile,gray):


    plt.figure('color')

    plt.imshow(gray, cmap='gray')

    plt.axis('off')
    plt.show()



def savaGray(outputfile,gray):
    gray.save(outputfile)


def main():
    #需要进行灰度转换的原彩色图片存储位置
    inputfile='color_.gif'

    #需要保存的转换后的图片名
    outputfile='gray.gif'

    img = Image.open(inputfile)

    gray = img.convert('L')

    #显示灰色图片
    showGray(inputfile,gray)

    #保存灰色图片
    savaGray(outputfile,gray)


if __name__ == '__main__':
    main()
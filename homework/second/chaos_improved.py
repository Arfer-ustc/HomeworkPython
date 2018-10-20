def getInput():

    input_x=eval(input('Please input the first number between 0 to 1:'))
    input_y=eval(input('Please input the second number between 0 to 1:'))
    return input_x,input_y

def chaos_output():
    input_x,input_y=getInput()
    write_to_local(input_x,input_y)

    print('\nindex\t%.2f\t\t\t%.2f' %(input_x,input_y))
    print('-------------------------------')
    for i in range(10):
        input_x=3.9*input_x*(1-input_x)
        input_y=3.9*input_y*(1-input_y)
        print('%d\t\t%.6f\t\t%.6f' %(i+1,input_x,input_y))

def write_to_local(x,y):
    result='\nindex\t%.2f\t\t\t%.2f' %(x,y)
    f.write(result+'\n')
    line='-------------------------------'
    f.write(line+'\n')
    for i in range(10):
        x = 3.9 * x * (1 - x)
        y = 3.9 * y * (1 - y)
        line='%d\t\t%.6f\t\t%.6f' % (i + 1, x, y)
        f.write(line+'\n')


if __name__ == '__main__':
    f = open('result.txt', 'a')
    chaos_output()

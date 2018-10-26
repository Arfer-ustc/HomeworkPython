import math
class Student():
    def __init__(self,name,hours,qpoints):
        self.name = name
        self.hours = float(hours)
        self.qpoints = float(qpoints)

    def getName(self): return self.name

    def getHours(self): return self.hours

    def getQPoints(self): return self.qpoints

    def gpa(self):
        return self.qpoints / self.hours


def makeStudent(infoStr):
    # infoStr is a tab-separated line: name hours qpoints
    # returns a corresponding Student object
    name, hours, qpoints = infoStr.split("\t")
    return Student(name, hours, qpoints)


def showBest(gap,bestList):
    for stu in bestList:
        if stu.gpa()==gap:
            # print information about the best student
            print("The best student is: ", stu.getName())
            print("hours: ", stu.getHours())
            print("GPA: ", stu.gpa())


def main():
    #open the input file for reading
    filename=input( "Enter the name of the grade file: ")
    infile = open(filename, 'r')
    # set best to the record for the first student in the file
    best =makeStudent (infile . readline ())
    # process subsequent lines of the file
    studlist=[]
    for line in infile:
        # turn the line into a student record
        s = makeStudent(line)
        studlist.append(s)
        # if this student is best so far , remember it .
        if s.gpa() > best.gpa():
            best = s
            bestGpa=best.gpa()
    infile.close()
    showBest(best.gpa(),studlist)




if __name__ == '__main__':
    main()


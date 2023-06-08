"""
Created on Oct 19 01:28:28 2021
Contains basics machine learning statistics and csv readers programs 
mainly this file contain programs for basic pre processing
@author: ANIKET YADAV
"""

# nessesary imports
import csv


# block of program for reading csv file
class load_csv():
    """
    class contain many types of methods which use to read data.
    by load_csv([PATH_OF_FILE]).dataset(column, upto[row]) also read dataset and seperate feature and lable
    through load_csv([PATH_OF_FILE]).featured_dataset(upto[rows]), can also use top function(upto), and tail(upto) function
    rows upto you are selected
    """
    # specifying path by init
    def __init__(self, file_path):
        self.file = file_path

    # display dataset by two arguments column and upto
    # column = all ----- for represent all the column take's only all, and other's integer
    # upto = all ------- for representing all the data of a csv file, row also take's only all, and other's integer
    def dataset(self, column='all', upto='all'):
        '''read dataset with column number and upto rows number by default it's set on the all column 
        and all rows, but we can use number of column and rows'''
        try:
            # step to store data in row list variable
            rows = []
            # creating a function for converting string to floating point for allowing machine readable
            def float_type(point):
                floats = []
                try:
                    # converting each itration to floats
                    for i in point:
                        floats.append(float(i))
                except ValueError:
                    # except pass value error 
                    pass
                return floats

            # float converting each point not including j-dimension
            def float_type_each(point):
                try:
                    floats = float(point)
                    return floats
                except ValueError:
                    pass
                

            # open file and manipulate as csvf instance
            with open(self.file, 'r') as csvf:
                for row in csv.reader(csvf):
                    # check if all column is se
                    # print all row 
                    if column == 'all':
                        rows.append([i for i in float_type(row[:])])
                    else:
                        # else print a single particular column
                        rows.append(float_type_each(row[column]))
                # close the csv file
                csvf.close()
            # check if upto selected all
            # it's return all row
            if upto == 'all':
                return rows[1:]
            else:
                # else return row upto, where you are selected
                return rows[1:upto+1]
        except IndexError:
            # return error if you are selected wrong column
            return f"column {column} not exist in your dataset"
    
    def featured_dataset(self, upto='all'):
        '''featured dataset is allow to split features and lable returns two value one is feature and second is
        lable with arguments of rows'''
        try:
            # making empty to store feature in rows and lables in lable list
            feature = []
            lable = []
            def float_type(point):
                floats = []
                try:
                    for i in point:
                        floats.append(float(i))
                except ValueError:
                    pass
                return floats

            # open file and manipulate as csvf instance
            with open(self.file, 'r') as csvf:
                for row in csv.reader(csvf):
                     
                    # print all row 
                    # select all column except last most..
                    feature.append([i for i in float_type(row[:-1])])
                    # last column is selected as lables
                    lable.append(row[-1])
                # close the csv file
                csvf.close()
            # check if upto selected all
            # it's return all row
            if upto == 'all':
                return feature[1:], lable[1:]
            else:
                # else return row upto, where you are selected
                return feature[1:upto+1], lable[1:upto+1] 
        except IndexError:
            # return error if you are selected wrong column or row
            return "unexpected column/row in your dataset"

    def top(self, upto):
        '''function use for read top of the data upto desired rows'''
        select = self.dataset()
        return select[:upto]

    # this is same from above fuction but, it's use to read data rows from downward side (tail)
    def tail(self, upto):
        '''returns last most desired data points thsts, take argument as upto[rows]'''
        select = self.dataset()
        # -upto: is use for last point in list to be read upto ....
        return select[-upto:]

    def text_csv(self, upto='all'):
        '''To use for loading text data, if csv have text, character data which is exactly a string type
        load_csv(PATH_OF_FILE).text_csv(upto[rows])'''
        try:
            feature = []
            lable = []
            with open(self.file, 'r') as csvf:
                for row in csv.reader(csvf):
                    # print all row 
                    # select all column except last most..
                    feature.append(row[:-1])
                    # last column is selected as lables
                    lable.append(row[-1])
                # close the csv file
                csvf.close()
                # check if upto selected all
                # it's return all row
            if upto == 'all':
                return feature[1:], lable[1:]
            else:
                # else return row upto, where you are selected
                return feature[1:upto+1], lable[1:upto+1]
        except IndexError:
            # return error if you are selected wrong column or row
            return "unexpected column/row in your dataset"


# there are many types of loss functions which use for finding accuracy of our alorithems
# this is our goal to reduse error of our model by decrease losses, to getting the optimum value '*w' and '*b' 
class loss_function():
    '''Loss function is use to determine the error between predicted and actual labels,
    also for the machine to find the optimum value of 'w' and 'b' during training a set
    takes list of the feature and labels (respective x and y), and w and b (initialize with 0.0) '''
    # loding actual and predicted data rows as list form to initialize different types of losses
    def __init__(self, y, x, w=0.0, b=0.0):
        # initially define value of 'w' and 'b' which is 0.0 by default
        (self.y, self.x, self.w, self.b) = (y, x, w, b)
    
    # -----'N' is always define total numbers of data points in actual dataset##----- #

    # L2 loss is stands for mean square error (MSE)
    # returns a floating point of value
    def L2_loss(self):
        ''' L2 loss for finding the mean square error of a predicted list and
        actual list of labels'''
        error = 0.0
        N = len(self.x)
        for i in range(N):
            # increases error value by mean of sqare of predicted and actual data points
            error += (self.y[i] - (self.w*self.x[i]+self.b))**2
        return error/float(N)

    # this is only a local program to calculate difference of actual and predicted data points
    # and returns a list of errors 
    def absolute_error(self):
        errors = []
        N = len(self.x)
        for i in range(N):
            # directly substract the value of actual and predicted points
            errors.append((self.w*self.x[i]+self.b) - self.y[i])
        return errors
    
    # L1 loss is stands for mean absolute error (MAE)
    # returns floating point
    def L1_loss(self):
        errors = 0.0
        N = len(self.x)
        for i in range(N):
            # applying calculation upto range
            errors += self.y[i] - self.w*self.x[i]+self.b
        return errors/float(N)
   
   # SSE is nothing but sum of square error returns the list of errors
    def SSE(self):
        errors = []
        N = len(self.x)
        for i in range(N):
            # adding the prediction and actual of square
            errors.append((self.y[i] - (self.w*self.x[i]+self.b))**2)
        return errors

    # RMSD - root mean square, MSE is divided by total number of list of predictions
    # and after finding root of it's result
    def RMSD(self):
        N = len(self.x)
        # ---->>> **0.5 is use for finding 'root' <<<---- #
        return (self.L2_loss()/N)**0.5
    
    # the hinge loss is a loss function used for training classifiers. the hinge loss is
    # used for "maximum margin" classification, most notably for support vector machine (SVM)
    def hinge(self):
        hinge = []
        N = len(self.x)
        for i in range(N):
            hinge.append(max(0, 1 - (self.w*self.x[i]+self.b) * self.y[i]))
        # returns list of errors
        return hinge


# statistics use formulas that's find mean, median, variance and std. deviation
# takes a list of feature 'X' return output as float
# but in the case of median it's returns 
class statistics():
    """
    statistics class function is use for the performing basic task about, finding mean or the avg. value of set, variance,
    median and standard deviation.
    """
    def __init__(self, x_set):
        self.x = x_set
        self.N = len(x_set)

    def mean(self):
        Elements = sum(self.x)
        return Elements/float(self.N)

    def variance(self):
        mean = self.mean()
        sumation = 0.0
        for i in self.x:
            sumation += (i - mean)**2
        return sumation/self.N

    def median(self):
        if self.N % 2 == 0:
            term = int((self.N + 1)/2)
            return self.x[term-1]
        else:
            d = self.N/2
            term = int(d + (d +1)/2)
            return self.x[term-1]    
    
    def standard_dev(self):
        mean = self.mean()
        sumation = 0.0
        for i in self.x:
            sumation += (i - mean)**2
        return (sumation/self.N)**0.5


def covarince(x_set, y_set):
    x_mean = statistics(x_set).mean()
    y_mean = statistics(y_set).mean()
    addtion = 0.0
    n = len(x_set)
    for x, y in zip(x_set, y_set):
        addtion += (x - x_mean)*(y - y_mean)
    return addtion/n


# we need a way to determine if there is linear correlation or not, so we calculate what is know 
# as the PRODUCT-MOMENT CORRELATION COEFFICIENT.
def Product_moment_CC(x_set, y_set):
    std_y = statistics(y_set).standard_dev()
    std_x = statistics(x_set).standard_dev()
    x_mean = statistics(x_set).mean()
    y_mean = statistics(y_set).mean()
    addtion = 0.0
    n = len(x_set)
    for x, y in zip(x_set, y_set):
        addtion += (x - x_mean)*(y - y_mean)
    covar = addtion/n
    return covar/std_x*std_y


################################## END OF THE PROGRAM ###############################################
pd = [12.3, 53.2, 52.2, 13.4, 83.5]
at = [26.7, 34.2, 52.0, 63.5, 12.6]
a = [0,1,1,0,1,1,1,0,0,1]
p = [0,0,0,1,0,1,1,0,1,0]
set = [3, 6, 9, 2, 7]
# print(loss_function(pd, at).SSE())
# print(covarince(at, pd))
# print(load_csv('A:/BIOINFORMAICS/Machine Learning/Machine learning/docs/dicisionDataset.csv').text_csv())
# print(load_csv('A:/BIOINFORMAICS/Machine Learning/Machine learning/docs/dicisionDataset.csv').featured_dataset())
# print(statistics(pd).mean())
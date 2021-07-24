import numpy as np
import pandas as pd
import matplotlib
import csv
from numpy import genfromtxt

import math
import ast
# Global variables
phase = "train"  # phase can be set to either "train" or "eval"

""" 
You are allowed to change the names of function arguments as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.

"""
def get_test(file_path):
    my_data = genfromtxt(file_path, delimiter = ',', dtype = float)
    phi = my_data[1:,1:6]
    data=pd.read_csv(file_path)
    date=np.array(pd.to_datetime(data['pickup_datetime']).dt.year)
    month=np.array(pd.to_datetime(data['pickup_datetime']).dt.month)
    day=np.array(pd.to_datetime(data['pickup_datetime']).dt.day)
    dist = np.power(np.power(data['pickup_longitude'] - data['dropoff_longitude'],2) + np.power(data['pickup_latitude'] - data['dropoff_latitude'],2),0.5)
    ones = np.ones([phi.shape[0],1])
    
    phi = np.column_stack((date,phi))
    phi = np.column_stack((month,phi))
    phi = np.column_stack((day,phi))
    phi = np.column_stack((dist,phi))
    phi = np.column_stack((np.power(dist,3),phi))
    phi = np.column_stack((np.power(dist,5),phi))
    phi = np.column_stack((np.power(dist,7),phi))
    phi = (phi - np.mean(phi,axis=0))/np.std(phi,axis=0)
    phi = np.column_stack((ones,phi))

    return phi
    
    
def get_test_1(file_path):
    my_data = genfromtxt(file_path, delimiter = ',', dtype = float)
    phi = my_data[1:,1:6]
    data=pd.read_csv(file_path)
    date=np.array(pd.to_datetime(data['pickup_datetime']).dt.year)
    month=np.array(pd.to_datetime(data['pickup_datetime']).dt.month)
    day=np.array(pd.to_datetime(data['pickup_datetime']).dt.day)
    dist = np.power(np.power(data['pickup_longitude'] - data['dropoff_longitude'],2) + np.power(data['pickup_latitude'] - data['dropoff_latitude'],2),0.5)
    ones = np.ones([phi.shape[0],1])
    
    phi = np.column_stack((dist,phi))
    phi = np.column_stack((np.power(dist,3),phi))
    phi = np.column_stack((np.power(dist,5),phi))
    phi = np.column_stack((np.power(dist,7),phi))
    phi = (phi - np.mean(phi,axis=0))/np.std(phi,axis=0)
    phi = np.column_stack((ones,phi))

    return phi

def get_test_2(file_path):
    my_data = genfromtxt(file_path, delimiter = ',', dtype = float)
    phi = my_data[1:,1:6]
    data=pd.read_csv(file_path)
    date=np.array(pd.to_datetime(data['pickup_datetime']).dt.year)
    month=np.array(pd.to_datetime(data['pickup_datetime']).dt.month)
    day=np.array(pd.to_datetime(data['pickup_datetime']).dt.day)
    dist = np.power(np.power(data['pickup_longitude'] - data['dropoff_longitude'],2) + np.power(data['pickup_latitude'] - data['dropoff_latitude'],2),0.5)
    ones = np.ones([phi.shape[0],1])
    
    phi = np.column_stack((date,phi))
    phi = np.column_stack((month,phi))
    phi = np.column_stack((day,phi))
    phi = np.column_stack((dist,phi))
    phi = np.column_stack((np.power(dist,3),phi))
    phi = np.column_stack((np.power(dist,5),phi))
    phi = np.column_stack((np.power(dist,7),phi))
    phi = np.column_stack((np.power(dist,7),phi))
    phi = np.column_stack((np.power(dist,9),phi))
    phi = np.column_stack((np.power(dist,11),phi))
    phi = np.column_stack((np.power(dist,13),phi))
    phi = np.column_stack((np.power(dist,15),phi))
    phi = (phi - np.mean(phi,axis=0))/np.std(phi,axis=0)
    phi = np.column_stack((ones,phi))

    return phi

def get_features(file_path):
    my_data = genfromtxt(file_path, delimiter = ',', dtype = float)
    phi = my_data[1:,1:6]
    data=pd.read_csv(file_path)
    date=np.array(pd.to_datetime(data['pickup_datetime']).dt.year)
    month=np.array(pd.to_datetime(data['pickup_datetime']).dt.month)
    day=np.array(pd.to_datetime(data['pickup_datetime']).dt.day)
    dist = np.power(np.power(data['pickup_longitude'] - data['dropoff_longitude'],2) + np.power(data['pickup_latitude'] - data['dropoff_latitude'],2),0.5)
    ones = np.ones([phi.shape[0],1])
    
    phi = np.column_stack((date,phi))
    phi = np.column_stack((month,phi))
    phi = np.column_stack((day,phi))
    phi = (phi - np.mean(phi,axis=0))/np.std(phi,axis=0)
    phi = np.column_stack((ones,phi))

    my_data = genfromtxt(file_path, delimiter = ',', dtype = float)    
    y = my_data[1:,6]
    y = y.reshape((phi.shape[0],1))

    # Given a file path , return feature matrix and target labels 
    
    
    return phi, y

def get_features_basis1(file_path):
    # Given a file path , return feature matrix and target labels 
    my_data = genfromtxt(file_path, delimiter = ',', dtype = float)
    phi = my_data[1:,1:6]
    data=pd.read_csv(file_path)
    date=np.array(pd.to_datetime(data['pickup_datetime']).dt.year)
    month=np.array(pd.to_datetime(data['pickup_datetime']).dt.month)
    day=np.array(pd.to_datetime(data['pickup_datetime']).dt.day)
    dist = np.power(np.power(data['pickup_longitude'] - data['dropoff_longitude'],2) + np.power(data['pickup_latitude'] - data['dropoff_latitude'],2),0.5)
    ones = np.ones([phi.shape[0],1])
    
    phi = np.column_stack((dist,phi))
    phi = np.column_stack((np.power(dist,3),phi))
    phi = np.column_stack((np.power(dist,5),phi))
    phi = np.column_stack((np.power(dist,7),phi))
    phi = (phi - np.mean(phi,axis=0))/np.std(phi,axis=0)
    phi = np.column_stack((ones,phi))

    my_data = genfromtxt(file_path, delimiter = ',', dtype = float)
    y = my_data[1:,6]
    y = y.reshape((phi.shape[0],1))
 
    
    return phi, y

def get_features_basis2(file_path):
    # Given a file path , return feature matrix and target labels 
    my_data = genfromtxt(file_path, delimiter = ',', dtype = float)
    phi = my_data[1:,1:6]
    data=pd.read_csv(file_path)
    date=np.array(pd.to_datetime(data['pickup_datetime']).dt.year)
    month=np.array(pd.to_datetime(data['pickup_datetime']).dt.month)
    day=np.array(pd.to_datetime(data['pickup_datetime']).dt.day)
    dist = np.power(np.power(data['pickup_longitude'] - data['dropoff_longitude'],2) + np.power(data['pickup_latitude'] - data['dropoff_latitude'],2),0.5)
    ones = np.ones([phi.shape[0],1])
    
    phi = np.column_stack((date,phi))
    phi = np.column_stack((month,phi))
    phi = np.column_stack((day,phi))
    phi = np.column_stack((dist,phi))
    phi = np.column_stack((np.power(dist,3),phi))
    phi = np.column_stack((np.power(dist,5),phi))
    phi = np.column_stack((np.power(dist,7),phi))
    phi = np.column_stack((np.power(dist,7),phi))
    phi = np.column_stack((np.power(dist,9),phi))
    phi = np.column_stack((np.power(dist,11),phi))
    phi = np.column_stack((np.power(dist,13),phi))
    phi = np.column_stack((np.power(dist,15),phi))
    phi = (phi - np.mean(phi,axis=0))/np.std(phi,axis=0)
    phi = np.column_stack((ones,phi))

   
   
    my_data = genfromtxt(file_path, delimiter = ',', dtype = float)
    
    y = my_data[1:,6]
    y = y.reshape((phi.shape[0],1))

    
    return phi, y

def compute_RMSE(phi, w , y) :
    
    error = np.sum(np.square(np.matmul(phi,w)-y))
    error=error/phi.shape[0]
    error = math.sqrt(error)
    # Root Mean Squared Error
    
    return error

def generate_output(phi_test, w):
    y_test = (np.matmul(phi_test,w))
    y_sub = np.zeros((20000,2))
    #y_sub[0,0] = 'Id'
    #y_sub[0,1] = 'fare'
    for i in range(1,20001):
            y_sub[i-1,0] = i-1
            y_sub[i-1,1] = y_test[i-1]
    np.savetxt("submission.csv",y_sub,fmt='%g', delimiter = ',', header="Id,fare", comments="")        

    # writes a file (output.csv) containing target variables in required format for Kaggle Submission.
    
def closed_soln(phi, y):
    # Function returns the solution w for Xw=y.
     return np.linalg.pinv(phi).dot(y)
    
def gradient_descent(phi, y) :
    w = np.zeros([phi.shape[1],1])
    alpha =0.01
    num_iter = 5000
    for i in range(1,num_iter):
        gd1 = np.matmul(np.transpose(phi),np.matmul(phi,w))
        gd2 = np.matmul(np.transpose(phi),y)
        gd= gd1-gd2
     #print(gd.shape)
        w = w -(alpha/phi.shape[0])*gd


    # Mean Squared Error

    return w

def sgd(phi, y) :
    # Mean Squared Error
    w = np.zeros([phi.shape[1],1])
    alpha =16
    num_iter = 60000
    for i in range(1,num_iter):
        rand = np.random.randint(0,phi.shape[0])
        x_i = phi[rand,None]
        y_i = y[rand,None]
        #print(y_i.shape)
        gd1 = np.matmul(np.transpose(x_i),np.matmul(x_i,w))
        gd2 = np.matmul(np.transpose(x_i),y_i)
        gd= gd1-gd2
     #print(gd.shape)
        w = w -(alpha/phi.shape[0])*gd


    return w


def pnorm(phi, y, p) :
    # Mean Squared Error
    w = np.zeros([phi.shape[1],1])
    alpha =0.01
    num_iter = 1000
    l = 0.1
    for i in range(1,num_iter):
        gd1 = np.matmul(np.transpose(phi),np.matmul(phi,w))
        gd2 = np.matmul(np.transpose(phi),y)
        gd= gd1-gd2+l*p*(np.power(w,p-1))
     #print(gd.shape)
        w = w -(alpha/phi.shape[0])*gd


    return w    

    
def main():

#The following steps will be run in sequence by the autograder.

       ######## Task 1 #########
        phase = "train"
        phi, y = get_features('train.csv')
        w1 = closed_soln(phi, y)
        w2 = gradient_descent(phi, y)
        phase = "eval"
        phi_dev, y_dev = get_features('dev.csv')
        r1 = compute_RMSE(phi_dev, w1, y_dev)
        r2 = compute_RMSE(phi_dev, w2, y_dev)
        print('1a: ')
        print(abs(r1-r2))
        w3 = sgd(phi, y)
        r3 = compute_RMSE(phi_dev, w3, y_dev)
        print('1c: ')
        print(abs(r2-r3))

        ######## Task 2 #########
        w_p2 = pnorm(phi, y, 2)  
        w_p4 = pnorm(phi, y, 4)  
        r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
        r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
        print('2: pnorm2')
        print(r_p2)
        print('2: pnorm4')
        print(r_p4)

        ######## Task 3 #########
        phase = "train"
        phi1, y = get_features_basis1('train.csv')
        phi2, y = get_features_basis2('train.csv')
        phase = "eval"
        phi1_dev, y_dev = get_features_basis1('dev.csv')
        phi2_dev, y_dev = get_features_basis2('dev.csv')
        w_basis1 = pnorm(phi1, y, 2)  
        w_basis2 = pnorm(phi2, y, 2)  
        rmse_basis1 = compute_RMSE(phi1_dev, w_basis1, y_dev)
        rmse_basis2 = compute_RMSE(phi2_dev, w_basis2, y_dev)
        print('Task 3: basis1')
        print(rmse_basis1)
        print('Task 3: basis2')
        print(rmse_basis2)
        phi_test = get_test_2('test.csv') 
        generate_output(phi_test,w_basis2)
main()

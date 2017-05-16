
"""Cd into the spark directory and Run as ./bin/spark-submit cluster.py """
from __future__ import print_function
from csvsort import csvsort
# $example on$
from numpy import array
from math import sqrt
import numpy as np


# $example off$

from pyspark import SparkContext
import ast

# $example on$
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import *
import csv
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="KMeansExample")  # SparkContext
    sqlCtx = SQLContext(sc)
    
    # Load and parse the data
    #data = sc.textFile("/Users/studies/Desktop/ipl/batsman_stat.csv")
    #data = sc.textFile("/Users/studies/Desktop/ipl/bowler_stat.csv")
    data = sc.textFile("/Users/studies/Desktop/ipl/bowler_batsman_stat.csv")

    #parsedData = data.map(lambda line: line.split(',')).map(lambda x:array([ast.literal_eval(x[1]),ast.literal_eval(x[2]),ast.literal_eval(x[3]),ast.literal_eval(x[4]),ast.literal_eval(x[5])])) #bowler_stat
    #parsedData = data.map(lambda line: line.split(',')).map(lambda x: array([ast.literal_eval(x[1]),ast.literal_eval(x[2]),ast.literal_eval(x[3]),ast.literal_eval(x[4]),ast.literal_eval(x[5]),ast.literal_eval(x[6])])) #batsman_stat
    parsedData = data.map(lambda line: line.split(','))
    parsedData = parsedData.map(lambda x: array([ast.literal_eval(x[8]),ast.literal_eval(x[9]),ast.literal_eval(x[10])])) #bowler_batsman_stat
    
    # Build the model (cluster the data)
    clusters = KMeans.train(parsedData, 10, maxIterations=10000, runs=10)
    #print(clusters.clusterCenters)
    # Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return center

    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    #print("Within Set Sum of Squared Error =---------------------------------------------", str(WSSSE))

    # Save and load model
    clusters.save(sc, "/Users/studies/Desktop/ipl/output_bowler_batsman") #delete the existing directory and make new one for every csv file
    #clusters.save(sc, "/Users/studies/Desktop/ipl/output_batsman")
    #clusters.save(sc, "/Users/studies/Desktop/ipl/output_bowler")
    #sameModel = KMeansModel.load(sc, "/Users/studies/Desktop/ipl/output_bowler")
    #sameModel = KMeansModel.load(sc, "/Users/studies/Desktop/ipl/output_batsman")
    sameModel = KMeansModel.load(sc, "/Users/studies/Desktop/ipl/output_bowler_batsman")
    #with open("/Users/studies/Desktop/ipl/bowler_stat.csv","r")as fd:
    #with open("/Users/studies/Desktop/ipl/batsman_stat.csv","r")as fd:
    with open("/Users/studies/Desktop/ipl/bowler_batsman_stat.csv","r")as fd:
        rdr=csv.reader(fd,delimiter=",")
        next(rdr, None)
        #with open("/Users/studies/Desktop/ipl/output_bowler/Results_Bowler.csv","wb")as f:
        #with open("/Users/studies/Desktop/ipl/output_batsman/Results_Batsmen.csv","wb")as f:
        with open("/Users/studies/Desktop/ipl/output_bowler_batsman/Results_Bowler_Batsmen.csv","wb")as f:
            for row in rdr:
                wrt=csv.writer(f,delimiter=",")
                li = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10]] #bowler_batsman_stat
                li.append(sameModel.predict([row[8],row[9],row[10]]))
                #li = [row[0],row[1],row[2],row[3],row[4],row[5],row[6]]  #batsman_stat
                #li.append(sameModel.predict([ast.literal_eval(row[1]),ast.literal_eval(row[2]),ast.literal_eval(row[3]),ast.literal_eval(row[4]),ast.literal_eval(row[5]),ast.literal_eval(row[6])]))
                #li = [row[0],row[1],row[2],row[3],row[4],row[5]]  #bowler_stat
                #li.append(sameModel.predict([ast.literal_eval(row[1]),ast.literal_eval(row[2]),ast.literal_eval(row[3]),ast.literal_eval(row[4]),ast.literal_eval(row[5])]))
                #wrt.writerow(["Batsmen","Cluster"])
                wrt.writerow(li)
    
    #csvsort("/Users/studies/Desktop/ipl/output_bowler/Results_Bowler.csv",[1])
    #csvsort("/Users/studies/Desktop/ipl/output_batsman/Results_Batsmen.csv",[1])
    #csvsort("/Users/studies/Desktop/ipl/output_bowler_batsman/Results_Bowler_Batsman.csv",[1])
    #print("RESULT:","-"*100,sameModel.predict([78,2012,122.01,1649,25.79]))
    print("Cluster centers----------------------------",sameModel.clusterCenters)
    # $example off$

    sc.stop()

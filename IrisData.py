# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 07:12:57 2019

@author: Aaditya
"""
#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing the dataset
dataset=pd.read_excel('Iris.xlsx')
dataset.drop('Unnamed: 5',axis=1)
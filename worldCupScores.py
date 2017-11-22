# 6.439 final project: predicting the winner of the World Cup 2018
# Jayson Vavrek et al, 2017

import sys
import csv
import math
import time
import numpy as np
import scipy as sp
import scipy.misc
import pandas as pd
import matplotlib.pyplot as plt

dropboxDir = '~/Dropbox (MIT)/Class Project/Project Data/rawdata/'

# Dictionary for year : location string
hostDict = {2014:'brazil', 2010:'southafrica', 2006:'germany', 2002:'koreajapan', 1998:'france'}

# Function to read data
def read_year_data(year):
  data = pd.read_csv(dropboxDir+'%s'%hostDict[year]+'%d'%year+'.csv')
  return data


def read_match_data():
  data = pd.read_csv(dropboxDir+'all_match_outcomes.csv')
  return data


# In the independent Poisson model, we can directly sample two Poisson distributions.
def scores_indep_poisson(lambda1, lambda2):
  return np.random.poisson([lambda1, lambda2])


# In the bivariate Poisson model, we need to build the joint pdf ourselves
def prob_bivariate_poisson(lambda0, lambda1, lambda2, x, y):
  prefactor1 = np.exp(-(lambda0+lambda1+lambda2))
  prefactor2 = np.power(lambda1,x) * np.power(lambda2,y) / (1.0 * np.math.factorial(x) * np.math.factorial(y))
  sumfactor = 0.0
  for i in xrange(min(x,y)+1):
    sumfactor += ( sp.misc.comb(x,i) * sp.misc.comb(y,i) * np.math.factorial(i) * np.power(lambda0/(1.0*lambda1*lambda2),i) )
  return prefactor1 * prefactor2 * sumfactor


# FIXME unsure if this uses the marginal means or actual parameters
def prob_diff_bivariate_poisson():
  pass


# Build a table of BVP probabilities if necessary
def build_bivariate_poisson_table(lambda0, lambda1, lambda2, nmax=10):
  joint_prob = np.zeros([nmax,nmax])
  for x in xrange(nmax):
    for y in xrange(nmax):
      joint_prob[x,y] = prob_bivariate_poisson(lambda0, lambda1, lambda2, x, y)
  return joint_prob

# TODO:
# Understand different parameters better
# Use the BVP to predict the winner of a game
# Build a tournament structure table and propagate
# MC sample initial distribution

#t = build_bivariate_poisson_table(0.2,2.1,3.2)
#print t

data = read_year_data(1998)










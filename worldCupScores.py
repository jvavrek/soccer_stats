# 6.439 final project: predicting the winner of the World Cup 2018
# Jayson Vavrek et al, 2017

import sys
import numpy as np
import scipy as sp
import scipy.misc
import scipy.special
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from project_functions import Create_Feature_Matrix

np.set_printoptions(precision=3, linewidth=180)

# Hardcoded (!) path to project data
dropboxDir = '~/Dropbox (MIT)/Class Project/Project Data/'

# Dictionary for year : location string
hostDict = {2014:'brazil', 2010:'southafrica', 2006:'germany', 2002:'koreajapan', 1998:'france'}

# Function to read data set for the World Cup of a given year
def read_year_data(year):
  data = pd.read_csv(dropboxDir+'%s'%hostDict[year]+'%d'%year+'.csv')
  return data


# Function to read data set containing all the individual match data
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


# FIXME unsure if this uses the marginal means or actual parameters.
# Seems to use actual parameters
def prob_diff_bivariate_poisson(lambda1, lambda2, z):
  factor1 = np.exp(-(lambda1+lambda2))
  factor2 = np.pow((lambda1/lambda2),z/2.0)
  factor3 = sp.special.iv(z, 2.0*np.sqrt(lambda1*lambda2) )
  return factor1 * factor2 * factor3


# Build a table of BVP probabilities if necessary
# Convention is (Team1,Team2) is (row,col) despite using x,y notation below.
# I have validated this for default params against tabulated results in the BVP slides at
# http://www2.stat-athens.aueb.gr/~karlis/Bivariate%20Poisson%20Regression.pdf
def build_bivariate_poisson_table(lambda0=0.1, lambda1=1.0, lambda2=0.9, nmax=10):
  joint_prob = np.zeros([nmax,nmax])
  for x in xrange(nmax):
    for y in xrange(nmax):
      joint_prob[x,y] = prob_bivariate_poisson(lambda0, lambda1, lambda2, x, y)
  return joint_prob/np.sum(joint_prob)


# Sample from a table of BVP probabilities. Since numpy/scipy don't support
# randomly-sampling 2D pmfs, work some modulus magic. Reshaping every time
# might be slow, so could return array of samples or pass pre-shaped table.
def scores_bivariate_poisson(prob_table):
  n  = prob_table.shape[0]
  n2 = prob_table.size
  tab1d = prob_table.reshape(n2)
  sample = np.random.choice(range(n2), p=tab1d)
  s1 = sample//n
  s2 = sample%n
  print sample, s1, s2
  return s1, s2


# (Currently skeleton) code for various regressions.
# Ideally the way this should work is if scoreMatrix is nx2, this should regress on the BVP;
# if scoreMatrix is nx1 (for score _differences_), it should regress on the BVPD;
# should also be able to regress (for both cases) on the independent Poisson model
# This means the heavy lifting of slicing the dataframe should be done in another function.
#                  BVP   IP  BVPD
# scoreMatrix in   nx2  nx2   nx1
# parameters out     3    2     2
def score_regression(featureMatrix, scoreMatrix, opt='linear', alpha=0.5):
  reg = None
  if opt == 'linear':
    reg = linear_model.LinearRegression()
  elif opt == 'ridge':
    reg = linear_model.Ridge(alpha=alpha)
  elif opt == 'lasso':
    reg = linear_model.Lasso(alpha=alpha)
  else:
    print "Error: bad option %s"%opt

  reg.fit(featureMatrix, scoreMatrix) # NOTE can pass n_jobs parameter > 1 if too slow
  return reg


# Wrapper function to get the lambda parameters from the regression, breaking them apart.
# NOTE: The regression should return log(lambda). As a result, using the predict()
# method might not work.
def get_lambda_params(regression):
  params  = regression.get_params()
  nparams = len(params)

  if nparams == 2: # independent Poisson
    return np.exp(params[0]), np.exp(params[1])
  elif nparams == 3: # bivariate Poisson
    return np.exp(params[0]), np.exp(params[1]), np.exp(params[2])
  else:
    print "Error: bad number of parameters %d"%nparams
    return None


def generate_test_scores(lambda0, lambda1, lambda2):
  tab = build_bivariate_poisson_table(lambda0, lambda1, lambda2)
  print tab


# TODO:
# Understand different parameters better
# Use the BVP to predict the winner of a game
# Build a tournament structure table and propagate
# MC sample initial distribution
# Better error handling?

#t = build_bivariate_poisson_table(0.2,2.1,3.2)
#print t

#data = read_year_data(1998)
#data = read_match_data()

#generate_test_scores(0.1,1.0,0.9)
#tab = build_bivariate_poisson_table(lambda0=0.1, lambda1=1.0, lambda2=0.9, nmax=10)
#scores_bivariate_poisson(tab)


def build_matrices(diff=False):
  fm = Create_Feature_Matrix(dropboxDir, 
                             match_data_file_location = dropboxDir + 'all_match_outcomes.csv',
                             years = [2014,2010,2006,2002],
                             features_to_consider = ['Matches Played',
                                                     'Yellow_Per_Game_Avg', 
                                                     'YellowRed_Per_Game_Avg',
                                                     'Red_Per_Game_Avg', 
                                                     'Goal_Per_Game_Avg', 
                                                     'Goal_Against_Per_Game_Avg',
                                                     'PenGoal_Per_Game_Avg',
                                                     'FIFA rank',
                                                     'seed',
                                                     'host',
                                                     'stars',
                                                     'cohesion', 
                                                     'dist', 
                                                     'cohesion sans 1'])
  sm = None # FIXME left off here. Build score matrix; return differences of scores (home-away) if diff==True

  return fm, sm











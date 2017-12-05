# 6.439 final project: predicting the winner of the World Cup 2018
# Jayson Vavrek et al, 2017

# TODO:
# Use the BVP to predict the winner of a game
# Build a tournament structure table and propagate
# MC sample initial distribution
# Better error handling?
# Time profile?
# StandardScaler?

import sys
import numpy as np
import scipy as sp
import scipy.misc
import scipy.special
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.optimize import minimize
from project_functions import Create_Feature_Matrix

np.set_printoptions(precision=3, linewidth=180)

# Hardcoded (!) path to project data
dropboxDir = '~/Dropbox (MIT)/Class Project/Project Data/'

# Dictionary for year : location string
hostDict = {2014:'brazil', 2010:'southafrica', 2006:'germany', 2002:'koreajapan', 1998:'france'}

# Function to read data set for the World Cup of a given year.
# Superseded by Create_Feature_Matrix().
def read_year_data(year):
  data = pd.read_csv(dropboxDir+'%s'%hostDict[year]+'%d'%year+'.csv')
  return data


# Function to read data set containing all the individual match data.
# Superseded by Create_Feature_Matrix().
def read_match_data():
  data = pd.read_csv(dropboxDir+'all_match_outcomes.csv')
  return data


# In the independent Poisson model, we can directly sample two Poisson distributions.
def scores_indep_poisson(lambda1, lambda2):
  return np.random.poisson([lambda1, lambda2])


# In the bivariate Poisson model, we need to build the joint pdf ourselves
def prob_bivariate_poisson(lambda0, lambda1, lambda2, x, y):
  x,y = int(np.round(x)), int(np.round(y)) # recast just to be sure
  prefactor1 = np.exp(-(lambda0+lambda1+lambda2))
  prefactor2 = np.power(lambda1,x) * np.power(lambda2,y) / (1.0 * np.math.factorial(x) * np.math.factorial(y))
  sumfactor = 0.0
  for i in xrange(min(x,y)+1):
    sumfactor += ( sp.misc.comb(x,i) * sp.misc.comb(y,i) * np.math.factorial(i) * np.power(lambda0/(1.0*lambda1*lambda2),i) )
  return prefactor1 * prefactor2 * sumfactor


# FIXME unsure if this uses the marginal means or actual parameters.
# Seems to use actual parameters.
# Needs validation. (Don't we all...)
def prob_diff_bivariate_poisson(lambda1, lambda2, z):
  factor1 = np.exp(-(lambda1+lambda2))
  factor2 = np.pow((lambda1/lambda2),z/2.0)
  factor3 = sp.special.iv(z, 2.0*np.sqrt(lambda1*lambda2) )
  return factor1 * factor2 * factor3


# Build a table of BVP probabilities if necessary.
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


# Code for various _linear_ regressions. The heavy lifting of building the
# input dataframes is done in build_dataframes.
#   Input: features (nxp DataFrame), scores (nx2 or nx1 DataFrame), others
#   Output: reg (linear_model object)
def score_regression(features, scores, opt='linear', alpha=0.5):
  reg = None
  if opt == 'linear':
    reg = linear_model.LinearRegression()
  elif opt == 'ridge':
    reg = linear_model.Ridge(alpha=alpha)
  elif opt == 'lasso':
    reg = linear_model.Lasso(alpha=alpha)
  else:
    print "Error: bad option %s"%opt

  featureMatrix, scoreMatrix = features.values, scores.values

  reg.fit(featureMatrix, scoreMatrix)
  print "Created %s model"%opt
  print "  variance explained: %.2f"%reg.score(featureMatrix,scoreMatrix)
  return reg


# Implementation of the EM algorithm as specified in the original BVP pdf.
# Will need to think about this a bit more, since it requires a model for the
# lambda as a function of the beta. Probably best to choose this model based on
# the results of the linear/ridge/lasso regression as we discussed earlier.
def BVP_EM_algorithm(features, scores):
  n_obs, n_coef = features.shape
  svec, xvec, yvec = np.zeros(n_obs), np.zeros(n_obs), np.zeros(n_obs)
  # initial guesses for lambda, beta parameters
  betaMatrix = np.ones([3, n_coef])*1e-6 # initial guess; 3xn makes indexing easier
  betaMatrixOld = betaMatrix*1e6 # just to make sure it doesn't converge first loop
  results = None

  # Define the likelihood function: the probability of observing the data given the model.
  # We would like to MAXIMIZE the likelihood in the M-step of BVP_EM_algorithm(), so we
  # will MINIMIZE the negative log-likelihood. This is defined internally within BVP_EM_algorithm
  # so that it can access scores/features without needing to define any globals.
  # See https://stackoverflow.com/questions/7718034/maximum-likelihood-estimate-pseudocode
  def neg_log_likelihood(betas, modified_scores):
    betas = betas.reshape([3, n_coef]) # weird that this is necessary but OK
    nloglike = 0.0
    for i in xrange(n_obs):
      [xi,yi] = modified_scores.iloc[i].values
      log_lambdas = compute_log_lambdas(betas, features.iloc[i])
      lambdas = map(np.exp, log_lambdas)
      nloglike += -np.log(prob_bivariate_poisson(lambdas[0], lambdas[1], lambdas[2], xi, yi))
    return nloglike

  # Convergence test: normalized sum of squared residuals
  def betaConvergence(betaMatrixOld, betaMatrix):
    res = betaMatrix - betaMatrixOld
    ssq = np.sum(res**2)
    nsq = ssq/(1.0*res.size)
    return nsq

  # Compute the log_lambdas from reg coeffs betas and features wi of a single observation i.
  # Returns a vector of three log_lambdas.
  def compute_log_lambdas(betas, wi):
    wi = wi.values
    ll0, ll1, ll2 = np.dot(betas[0], wi), np.dot(betas[1], wi), np.dot(betas[2], wi)
    return np.array([ll0, ll1, ll2])

  # main EM loop
  k = 0
  converged = False
  while not converged:
    print '\nEM iteration %d'%k
    if k == 5:
      print '  aborting on k = %d'%k
      break

    # E-step
    for i in xrange(n_obs):
      si = 0
      [xi,yi] = scores[['Score1','Score2']].iloc[i].values # be explicit about Score1,2 since we modify DataFrame below
      if min(xi,yi) > 0:
        log_lambdas = compute_log_lambdas(betaMatrix, features.iloc[i])
        lambdas = map(np.exp, log_lambdas)
        num   = prob_bivariate_poisson(lambdas[0], lambdas[1], lambdas[2], xi-1, yi-1)
        denom = prob_bivariate_poisson(lambdas[0], lambdas[1], lambdas[2], xi,   yi)
        si = lambdas[0] * num / denom
      svec[i] = int(np.round(si)) # FIXME unsure about this

    # M-step
    scores['x-s'] = scores['Score1'] - svec
    scores['y-s'] = scores['Score2'] - svec
    scoresMod = scores[['x-s','y-s']]
    print '  beginning minimization'
    results = minimize(neg_log_likelihood, betaMatrix, args=scoresMod, method='nelder-mead', tol=1e-3)
    # 4.5 minutes for tol=1e-2, 3 s for 1e-1, wow. It seems like the initial guess is within a tol of > 1e-2
    # and it exits immediately, whereas the 1e-2 tol actually produces different results
    print '  finished minimization in M-step'
    betaMatrixOld = betaMatrix
    betaMatrix = results.x.reshape([3,n_coef])

    print 'current lambda list:'
    for i in xrange(n_obs):
      ll = compute_log_lambdas(betaMatrix, features.iloc[i])
      lambdas_final = map(np.exp, ll)
      [xi,yi] = scores[['Score1','Score2']].iloc[i].values
      print lambdas_final, xi, yi

    # Test for convergence
    nsq = betaConvergence(betaMatrixOld, betaMatrix)
    print '  normalized sum of squares of residuals:', nsq
    if nsq < 1e-7:
      converged = True
      print '  converged'
      break

    k += 1 # end main EM loop

  return results


# Build and slice the datasets for the score_regression()
# and BVP_EM_algorithm() functions.
standard_features = ['Matches Played',
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
                     'cohesion sans 1']

def build_dataframes(diff=False):
  m = Create_Feature_Matrix(dropboxDir, 
                            match_data_file_location = dropboxDir + 'all_match_outcomes.csv',
                            years = [2014,2010,2006,2002])
  m.dropna(inplace=True)
  features = m.loc[:, standard_features[0]+'_C1' : standard_features[-1]+'_C2']

  scores = None
  if diff == True:
    scores = m['Score_Diff']
  else:
    scores = m[['Score1','Score2']]

  return features, scores


# NOTE for later:
#                  BVP   IP  BVPD
# scoreMatrix in   nx2  nx2   nx1
# reg parameters     3    2     2


# Simulate the seeding for the 2018 WC. Takes a featureMatrix (for the time being)
# to obtain the seeds, then creates brackets based on those seeds. Top 4 (?) teams
# are 'protected' and don't play each other, rest are placed randomly.
# NOTE: real draw for 2018 takes place Dec 1! Though, they might not do it randomly:
# https://www.si.com/soccer/2017/11/14/world-cup-group-draw-pots-russia-2018
# Could use random vs deterministic to infer how much of an advantage it is to be
# one of the 'protected' seeds, especially the host nation.
def create_tournament_seeds(featureMatrix):
  pass


# Perform the full tournament simulation. Need to look into details, but this should
# involve group play followed by elimination rounds. For each matchup, use the model
# to predict the score, and advance the winning teams until we have a winner.
def simulate_tournament():
  pass

#tab = build_bivariate_poisson_table(lambda0=0.1, lambda1=1.0, lambda2=0.9, nmax=10)
#scores_bivariate_poisson(tab)

fm, sm = build_dataframes(diff=False)
reg = score_regression(fm, sm, opt='linear')
#reg = score_regression(fm, sm, opt='ridge', alpha=0.5)
#reg = score_regression(fm, sm, opt='lasso', alpha=0.5)
results = BVP_EM_algorithm(fm,sm)









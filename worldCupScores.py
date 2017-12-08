# 6.439 final project: predicting the winner of the World Cup 2018
# Jayson Vavrek et al, 2017

# TODO:
# Use the BVP to predict the winner of a game
# Build a tournament structure table and propagate
# MC sample initial distribution
# Validation: test and training set scores
# Variations in AIC due to starting point

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

np.set_printoptions(precision=3, linewidth=200)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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
  #betaMatrix = np.ones([3, n_coef])*1e-6 # initial guess; 3xn makes indexing easier
  betaMatrix = np.random.rand(3,n_coef)*1e-6
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

  def compute_AIC(results):
    nll_min = results.fun
    k = results.x.size
    print 'nll_min =', nll_min
    print 'k =', k
    return 2.0*(k + nll_min)

  # main EM loop
  k = 0
  converged = False
  while not converged:
    print '\nEM iteration %d'%k
    if k == 10:
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
    results = minimize(neg_log_likelihood, betaMatrix, args=scoresMod, method='BFGS', tol=1e-9)
    print '  finished minimization in M-step'
    print '  status:', results.message
    betaMatrixOld = betaMatrix
    betaMatrix = results.x.reshape([3,n_coef]) # <------ here is where we get the betaMatrix from results

    print 'current lambda list:'
    for i in xrange(n_obs):
      ll = compute_log_lambdas(betaMatrix, features.iloc[i])
      lambdas_final = map(np.exp, ll)
      [xi,yi] = scores[['Score1','Score2']].iloc[i].values
      print lambdas_final, xi, yi

    print 'current beta matrix:'
    print betaMatrix

    aic = compute_AIC(results)
    print 'current AIC:', aic

    # Test for convergence
    nsq = betaConvergence(betaMatrixOld, betaMatrix)
    print '  normalized sum of squares of residuals:', nsq
    if nsq < 1e-10:
      converged = True
      print '  converged'
      break

    k += 1 # end main EM loop

  #return results
  return betaMatrix, aic


# Compute the log_lambdas from reg coeffs betas and features wi of a single observation i.
# Returns a vector of three log_lambdas.
def compute_log_lambdas(betas, wi):
  if type(wi) is not np.ndarray:
    wi = wi.values
  ll0, ll1, ll2 = np.dot(betas[0], wi), np.dot(betas[1], wi), np.dot(betas[2], wi)
  return np.array([ll0, ll1, ll2])


def validation_scores(betaMatrix, features, scores):
  print 'Running validation...'
  n_obs, n_coef = features.shape

  correct_scores  = 0
  correct_winners = 0
  #fbinary = 0
  r2 = 0

  for i in xrange(n_obs):
    ll = compute_log_lambdas(betaMatrix, features.iloc[i])
    lambdas_final = map(np.exp, ll)
    x_pred, y_pred = lambdas_final[0] + lambdas_final[1], lambdas_final[0] + lambdas_final[2]
    [xi,yi] = scores[['Score1','Score2']].iloc[i].values

    if int(np.round(x_pred)) == xi and int(np.round(y_pred)) == yi:
      correct_scores += 1
    if ((x_pred > y_pred) and (xi > yi)) or ((x_pred < y_pred) and (xi < yi)):
      correct_winners += 1
    r2 += ((xi-x_pred)**2 + (yi-y_pred)**2)

    #[sx,sy] = features[['seed_C1','seed_C2']].iloc[i].values
    #if ((sx < sy) and (xi > yi)) or ((sy < sx) and (yi > xi)):
    #  fbinary +=1

  correct_scores  /= (1.0*n_obs)
  correct_winners /= (1.0*n_obs)
  #fbinary /= (1.0*n_obs)

  return correct_scores, correct_winners, r2


# Takes base_features and adds _C1 and _C2 to the strings. Returns the new list features_out.
# Anjian suggests most important features seem to be y/r per game, GA per game, FIFA rank, host, seed, cohesion, dist
def create_feature_list(base_features):
  f1 = base_features[:]
  f2 = base_features[:]
  f1[:] = [i + '_C1' for i in f1]
  f2[:] = [i + '_C2' for i in f2]
  return f1 + f2


# Build and slice the datasets for the score_regression() and BVP_EM_algorithm() functions.
def build_dataframes(base_features=['Goal_Per_Game_Avg']):
  m = Create_Feature_Matrix(dropboxDir, 
                            match_data_file_location = dropboxDir + 'all_match_outcomes.csv',
                            years = [2014,2010,2006,2002])
  m.dropna(inplace=True)
  #features = m.loc[:, standard_features[0]+'_C1' : standard_features[-1]+'_C2']
  #features = m[['Matches Played_C1', 'Goal_Per_Game_Avg_C1', 'FIFA rank_C1', 'Matches Played_C2', 'Goal_Per_Game_Avg_C2', 'FIFA rank_C2']]
  features_list = create_feature_list(base_features)
  features = m[features_list]
  #features.loc[:,'dist_C1'] *= 1e-6
  #features.loc[:,'dist_C2'] *= 1e-6

  scores = m[['Score1','Score2']]

  return features, scores


# Simulate the seeding for the 2018 WC. Takes a featureMatrix (for the time being)
# to obtain the seeds, then creates brackets based on those seeds. Top 4 (?) teams
# are 'protected' and don't play each other, rest are placed randomly.
# NOTE: real draw for 2018 takes place Dec 1! Though, they might not do it randomly:
# https://www.si.com/soccer/2017/11/14/world-cup-group-draw-pots-russia-2018
# Could use random vs deterministic to infer how much of an advantage it is to be
# one of the 'protected' seeds, especially the host nation.
def create_tournament_seeds():
  m = pd.read_csv(dropboxDir+'2018'+'.csv')
  m.dropna(inplace=True)


# Perform the full tournament simulation. Need to look into details, but this should
# involve group play followed by elimination rounds. For each matchup, use the model
# to predict the score, and advance the winning teams until we have a winner.
def simulate_tournament(betaMatrix):
  m = pd.read_csv(dropboxDir+'2018'+'.csv')
  m.dropna(inplace=True)

  m['Group GF'] = 0
  m['Group GA'] = 0
  m['Group W'] = 0
  m['Group L'] = 0
  m['Group T'] = 0
  m['R16 W'] = 0
  m['R16 idx'] = 0
  m['Qtr W'] = 0
  m['Qtr idx'] = 0
  m['Semi W'] = 0
  m['Final W'] = 0

  # Group play: everyone plays everyone in their pool
  groups = pd.unique(m['Group'])
  for g in groups:
    group_data = m[m['Group'] == g]
    teams = pd.unique(group_data['Country'])
    for i in xrange(len(teams)):
      for j in xrange(i+1,len(teams)):
        df = pd.DataFrame()
        wi = np.array([group_data['Seed'].iloc[i], group_data['Seed'].iloc[j]]) # FIXME temporary hack
        lambdas = map(np.exp, compute_log_lambdas(betaMatrix,wi))
        pt = build_bivariate_poisson_table(lambdas[0], lambdas[1], lambdas[2], nmax=10)
        s1, s2 = scores_bivariate_poisson(pt)
        print '%s %i | %s %i'%(teams[i], s1, teams[j], s2)
        m['Group GF'].loc[m['Country']==teams[i]] += s1  # FIXME there has got to be a better/faster way to do this...
        m['Group GA'].loc[m['Country']==teams[i]] += s2
        m['Group GF'].loc[m['Country']==teams[j]] += s2
        m['Group GA'].loc[m['Country']==teams[j]] += s1
        if s1 > s2:
          m['Group W'].loc[m['Country']==teams[i]] += 1
          m['Group L'].loc[m['Country']==teams[j]] += 1
        elif s2 > s1:
          m['Group L'].loc[m['Country']==teams[i]] += 1
          m['Group W'].loc[m['Country']==teams[j]] += 1
        else:
          m['Group T'].loc[m['Country']==teams[i]] += 1
          m['Group T'].loc[m['Country']==teams[j]] += 1

  m.sort_values(['Group', 'Group W', 'Group GF'], ascending=[True,False,False], inplace=True)
  m.reset_index(inplace=True)
  print m

  m_sixteen = m[m.index%4<=1]
  print m_sixteen

  # Elimination rounds
  # Only top two teams from each pool advance to round of 16, then single knockout.
  # R16: A1 plays B2, A2 plays B1, etc
  groups = pd.unique(m_sixteen['Group'])
  for g in xrange(0, len(groups)-1, 2):
    this_group = m_sixteen[m_sixteen['Group'] == groups[g]]
    chal_group = m_sixteen[m_sixteen['Group'] == groups[g+1]]

    for i in xrange(2):
      if i==0:
        hi = this_group.iloc[0]
        lo = chal_group.iloc[1]
      else:
        hi = chal_group.iloc[0]
        lo = this_group.iloc[1]
  
      wi = np.array([hi['Seed'], lo['Seed']]) # FIXME temporary hack
      lambdas = map(np.exp, compute_log_lambdas(betaMatrix,wi))
      pt = build_bivariate_poisson_table(lambdas[0], lambdas[1], lambdas[2], nmax=10)
      s1, s2 = scores_bivariate_poisson(pt)
      print '%s %i | %s %i'%(hi['Country'], s1, lo['Country'], s2)
      if s1 >= s2:
        m['R16 W'].loc[m['Country']==hi['Country']] += 1 # FIXME need to handle ties!!
      elif s1 < s2:
        m['R16 W'].loc[m['Country']==lo['Country']] += 1
      if i==1:
        m['R16 idx'].loc[m['Country']==hi['Country']] += 1
        m['R16 idx'].loc[m['Country']==lo['Country']] += 1

  m_quarter = m[m['R16 W'] > 0]


  print m_quarter
  # Quarter finals, eight teams remaining.
  # Winner of A/B plays counterpart from C/D; winner of B/A plays that from D/C
  # Elimination rounds
  # Only top two teams from each pool advance to round of 16, then single knockout.
  # R16: A1 plays B2, A2 plays B1, etc
  groups = pd.unique(m_quarter['Group'])
  for g in xrange(0, len(groups)-1, 2):
    abcd_group = m_quarter[m_quarter['Group'].isin(['A','B','C','D'])]
    efgh_group = m_quarter[m_quarter['Group'].isin(['D','E','F','G'])]

    for i in xrange(2):
      if i==0:
        hi = this_group.iloc[0]
        lo = chal_group.iloc[1]
      else:
        hi = chal_group.iloc[0]
        lo = this_group.iloc[1]
  
      wi = np.array([hi['Seed'], lo['Seed']]) # FIXME temporary hack
      lambdas = map(np.exp, compute_log_lambdas(betaMatrix,wi))
      pt = build_bivariate_poisson_table(lambdas[0], lambdas[1], lambdas[2], nmax=10)
      s1, s2 = scores_bivariate_poisson(pt)
      print '%s %i | %s %i'%(hi['Country'], s1, lo['Country'], s2)
      if s1 >= s2:
        m['R16 W'].loc[m['Country']==hi['Country']] += 1 # FIXME need to handle ties!!
      elif s1 < s2:
        m['R16 W'].loc[m['Country']==lo['Country']] += 1
      if i==1:
        m['R16 idx'].loc[m['Country']==hi['Country']] += 1
        m['R16 idx'].loc[m['Country']==lo['Country']] += 1

  m_quarter = m[m['R16 W'] > 0]



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
                     'cohesion sans 1',
                     'dist']


# Poisson model tests
#tab = build_bivariate_poisson_table(lambda0=0.1, lambda1=1.0, lambda2=0.9, nmax=10)
#scores_bivariate_poisson(tab)

# Regression
#reg = score_regression(fm, sm, opt='linear')
#reg = score_regression(fm, sm, opt='ridge', alpha=0.5)
#reg = score_regression(fm, sm, opt='lasso', alpha=0.5)

# Model selection
#fm, sm = build_dataframes(base_features=['FIFA rank'])
#betaMatrix, aic = BVP_EM_algorithm(fm,sm)
#aicc = aic + 2*betaMatrix.size*(betaMatrix.size+1)/(1.0*(len(sm)-betaMatrix.size-1)) # disclaimer: some assumptions don't hold for this
#cs, cw, r2 = validation_scores(betaMatrix, fm, sm)
#print 'AIC; AICc; k; exact score rate; winner rate; r2; parameters:'
#print '%.4f; %.4f; %i; %.4f; %.4f; %.4f;'%(aic, aicc, betaMatrix.size, cs, cw, r2), list(fm.columns)

betaMatrix = np.array([[-1.842, -3.753], [-0.01, 0.019], [0.022, -0.014]])
simulate_tournament(betaMatrix)







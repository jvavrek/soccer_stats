# 6.439 final project: predicting the winner of the World Cup 2018
# Jayson Vavrek et al, 2017

# TODO:
# MC sample initial distribution

import sys
import operator
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


# Unsure if this uses the marginal means or actual parameters.
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
# If break_ties=True, use prob_table to determine who is more likely to
# score one more goal, then add one to that team.
def scores_bivariate_poisson(prob_table, break_ties=False):
  n  = prob_table.shape[0]
  n2 = prob_table.size
  tab1d = prob_table.reshape(n2)
  sample = np.random.choice(range(n2), p=tab1d)
  s1 = sample//n
  s2 = sample%n
  if s1 == s2 and break_ties == True:
    if prob_table[s1+1,s2] > prob_table[s1,s2+1]:
      s1 += 1
    else:
      s2 += 1
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
      svec[i] = int(np.round(si)) # Unsure about this

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

  correct_scores  /= (1.0*n_obs)
  correct_winners /= (1.0*n_obs)

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
  features_list = create_feature_list(base_features)
  features = m[features_list]
  scores = m[['Score1','Score2']]

  return features, scores


# Simulate the seeding for the 2018 WC. If randomized=False, read in the true groups.
# If randomized=True, do the same, then randomize the group assignments.
def create_tournament_brackets(randomized=False):
  m = pd.read_csv(dropboxDir+'2018'+'.csv')
  m.dropna(inplace=True)
  m.drop(['Seed - AW'], axis=1, inplace=True)
  m.set_index('Country', inplace=True)

  if randomized == True:
    group_list = 4*['A','B','C','D','E','F','G','H']
    np.random.shuffle(group_list)
    m['Group'] = group_list

  return m


# Perform the full tournament simulation. Simulates group play followed by elimination
# rounds. For each matchup, use the base_features and betaMatrix to predict the score,
# and advance the winning teams until we have a winner.
def simulate_tournament(betaMatrix, base_features=['seed','host'], randomized=False):
  m = create_tournament_brackets(randomized=randomized)
  print 'm='
  print m

  m['Group GF'] = 0
  m['Group GA'] = 0
  m['Group W'] = 0
  m['Group L'] = 0
  m['Group T'] = 0
  m['R16 W'] = 0
  m['R16 brac'] = 0
  m['Qtr W'] = 0
  m['Qtr brac'] = 0
  m['Semi W'] = 0
  m['Final W'] = 0

  def build_wi(t1, t2, base_features):
    wi = []
    for ft in base_features:
      wi.append(t1[ft])
    for ft in base_features:
      wi.append(t2[ft])
    return np.array(wi)

  def simulate_match(t1, t2, wi, break_ties=False):
    lambdas = map(np.exp, compute_log_lambdas(betaMatrix,wi))
    pt = build_bivariate_poisson_table(lambdas[0], lambdas[1], lambdas[2], nmax=10)
    s1, s2 = scores_bivariate_poisson(pt, break_ties)
    print '%s %i | %s %i'%(t1.name, s1, t2.name, s2)
    return s1, s2

  # Group play: everyone plays everyone in their pool
  print '\nBeginning group play...'
  groups = pd.unique(m['Group'])
  for g in groups:
    group_data = m[m['Group'] == g]
    teams = pd.unique(group_data.index)
    for i in xrange(len(teams)):
      for j in xrange(i+1,len(teams)):
        t1 = group_data.iloc[i]
        t2 = group_data.iloc[j]
        wi = build_wi(t1, t2, base_features)
        s1, s2 = simulate_match(t1, t2, wi, break_ties=False)
        m['Group GF'].loc[t1.name] += s1
        m['Group GA'].loc[t1.name] += s2
        m['Group GF'].loc[t2.name] += s2
        m['Group GA'].loc[t2.name] += s1
        if s1 > s2:
          m['Group W'].loc[t1.name] += 1
          m['Group L'].loc[t2.name] += 1
        elif s2 > s1:
          m['Group L'].loc[t1.name] += 1
          m['Group W'].loc[t2.name] += 1
        else:
          m['Group T'].loc[t1.name] += 1
          m['Group T'].loc[t2.name] += 1

  m.sort_values(['Group', 'Group W', 'Group GF'], ascending=[True,False,False], inplace=True)
  m.reset_index(inplace=True)
  print m


  # ====== Elimination rounds ====== #
  # Only top two teams from each pool advance to round of 16, then single knockout.

  # R16: A1 plays B2, A2 plays B1, etc
  print '\nBeginning round of 16...'
  m_sixteen = m[m.index%4<=1]
  m_sixteen.set_index('Country', inplace=True)
  m.set_index('Country', inplace=True)
  groups = pd.unique(m_sixteen['Group'])
  for g in xrange(0, len(groups)-1, 2):
    this_group = m_sixteen[m_sixteen['Group'] == groups[g]]
    chal_group = m_sixteen[m_sixteen['Group'] == groups[g+1]]

    for i in [0,1]:
      if i==0:
        t1 = this_group.iloc[0]
        t2 = chal_group.iloc[1]
      else:
        t1 = chal_group.iloc[0]
        t2 = this_group.iloc[1]
  
      wi = build_wi(t1, t2, base_features)
      s1, s2 = simulate_match(t1, t2, wi, break_ties=True)
      if s1 > s2:
        m['R16 W'].loc[t1.name] += 1
      elif s1 < s2:
        m['R16 W'].loc[t2.name] += 1
      else:
        print 'TIE!'
      if i==1:
        m['R16 brac'].loc[t1.name] += 1
        m['R16 brac'].loc[t2.name] += 1


  print '\nBeginning quarter finals...'
  m_quarter = m[m['R16 W'] > 0]
  print m_quarter
  # Quarter finals, eight teams remaining.
  # Winner of A/B plays counterpart from C/D; winner of B/A plays that from D/C
  # Elimination rounds
  # Only top two teams from each pool advance to round of 16, then single knockout.
  # R16: A1 plays B2, A2 plays B1, etc
  groups = pd.unique(m_quarter['Group'])
  abcd_group = m_quarter[m_quarter['Group'].isin(['A','B','C','D'])]
  efgh_group = m_quarter[m_quarter['Group'].isin(['E','F','G','H'])]

  for g in [abcd_group, efgh_group]:
    for i in [0,1]:
      subgroup = g[g['R16 brac']==i]
      t1 = subgroup.iloc[0]
      t2 = subgroup.iloc[1]
      wi = build_wi(t1, t2, base_features)
      s1, s2 = simulate_match(t1, t2, wi, break_ties=True)
      if s1 > s2:
        m['Qtr W'].loc[t1.name] += 1
      elif s1 < s2:
        m['Qtr W'].loc[t2.name] += 1
      else:
        print 'TIE!'
      if i==1:
        m['Qtr brac'].loc[t1.name] += 1
        m['Qtr brac'].loc[t2.name] += 1


  # Semis
  print '\nBeginning semi finals...'
  m_semi = m[m['Qtr W'] > 0]
  for i in [0,1]:
    subgroup = m_semi[m_semi['Qtr brac']==i]
    t1 = subgroup.iloc[0]
    t2 = subgroup.iloc[1]
    wi = build_wi(t1, t2, base_features)
    s1, s2 = simulate_match(t1, t2, wi, break_ties=True)
    if s1 > s2:
      m['Semi W'].loc[t1.name] += 1
    elif s1 < s2:
      m['Semi W'].loc[t2.name] += 1
    else:
      print 'TIE!'

  print m_semi


  # FINALS!!
  print '\nBeginning finals...'
  m_finals = m[m['Semi W'] > 0]
  t1 = m_finals.iloc[0]
  t2 = m_finals.iloc[1]
  wi = build_wi(t1, t2, base_features)
  s1, s2 = simulate_match(t1, t2, wi, break_ties=True)
  if s1 > s2:
    m['Final W'].loc[t1.name] += 1
  elif s1 < s2:
    m['Final W'].loc[t2.name] += 1
  else:
    print 'TIE!' 

  print m_finals
  print m

  winner = m[m['Final W']==1].index[0]
  second = m[(m['Semi W']==1) & (m['Final W']==0)].index[0]
  semis_list = m[m['Qtr W']==1].index.values

  print 'Winner:', winner
  print 'Second:', second
  print 'Semis:', semis_list

  return m, winner, semis_list


def MC_sample_tournament(betaMatrix, base_features=['seed','host'], randomized=False, trials=100):
  fname_w = 'winners'
  fname_s = 'semis'
  if randomized == True:
    fname_w += '_rand'
    fname_s += '_rand'
  else:
    fname_w += '_fixed'
    fname_s += '_fixed'
  fname_w += '.txt'
  fname_s += '.txt'

  fw = open(fname_w,'w')
  fs = open(fname_s,'w')

  for i in xrange(trials):
    print 'TRIAL %i'%i
    m, winner, semis_list = simulate_tournament(betaMatrix, base_features, randomized)
    fw.write(winner+'\n')
    for s in semis_list:
      fs.write(s+'\n')

  fw.close()
  fs.close()
  print 'Files %s, %s written'%(fname_w, fname_s)


def analyze_MC_samples(team_list, file_w = 'results/winners_fixed_seed.txt', file_s = 'results/semis_fixed_seed.txt'):
  lw = open(file_w).read()
  dw = {}
  for team in team_list:
    countw = lw.count(team)
    dw[team] = countw
  dw = sorted(dw.items(), key=operator.itemgetter(0), reverse=False)
  print 'winners, from %s:'%file_w
  print dw
  print

  ls = open(file_s).read()
  ds = {}
  for team in team_list:
    counts = ls.count(team)
    ds[team] = counts
  ds = sorted(ds.items(), key=operator.itemgetter(0), reverse=False)
  print 'semis, from %s:'%file_s
  print ds

  return dw


def generate_plots(dw_fixed, dw_random):
  win_frac_fixed  = [i[1] for i in dw_fixed]
  win_frac_random = [i[1] for i in dw_random]
  labels = [i[0] for i in dw_random]
  nums = range(len(dw_fixed))

  plt.bar(nums, win_frac_fixed,  color='b', alpha=0.5, width=1.0, align='center')
  plt.bar(nums, win_frac_random, color='r', alpha=0.5, width=1.0, align='center')
  plt.legend(['true seeding','randomized seeding'])
  plt.xticks(nums, [x[0] for x in dw_fixed], rotation='vertical')
  plt.ylabel('World Cup 2018 wins in 100 simulations')
  plt.xlim([-0.5,len(dw_fixed)-0.5])
  plt.subplots_adjust(bottom=0.21)
  plt.show(block=False)

# screw it, make some globals
countries_2018 = ['Uruguay', 'Egypt', 'Russia', 'Saudi Arabia', 'Portugal',
                  'Morocco', 'Spain', 'Iran', 'France', 'Denmark', 'Peru',
                  'Australia', 'Argentina', 'Iceland', 'Nigeria', 'Croatia',
                  'Switzerland', 'Brazil', 'Costa Rica', 'Serbia', 'Germany',
                  'Mexico', 'Sweden', 'Korea Republic', 'Belgium', 'Panama',
                  'England', 'Tunisia', 'Poland', 'Senegal', 'Colombia', 'Japan']

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
#fm, sm = build_dataframes(base_features=['seed','host'])
#betaMatrix, aic = BVP_EM_algorithm(fm,sm)
#aicc = aic + 2*betaMatrix.size*(betaMatrix.size+1)/(1.0*(len(sm)-betaMatrix.size-1)) # disclaimer: some assumptions don't hold for this
#cs, cw, r2 = validation_scores(betaMatrix, fm, sm)
#print 'AIC; AICc; k; exact score rate; winner rate; r2; parameters:'
#print '%.4f; %.4f; %i; %.4f; %.4f; %.4f;'%(aic, aicc, betaMatrix.size, cs, cw, r2), list(fm.columns)

betaMatrix_seed = np.array([[-1.94, -1.677],
                            [-0.016, 0.028],
                            [0.041, -0.025]])

betaMatrix_seedhost = np.array([[ -1.982e+00,   2.105e+01,  -1.737e+00,   1.336e+01],
                                [ -1.835e-02,  -3.072e-01,   2.987e-02,  -2.531e+02],
                                [  3.602e-02,  -2.729e+02,  -1.959e-02,   9.153e-02]])
#m, winner, semis_list = simulate_tournament(betaMatrix_seedhost,['seed','host'], randomized=False)
#MC_sample_tournament(betaMatrix_seedhost, base_features=['seed','host'], randomized=True, trials=100)

dw_fixed  = analyze_MC_samples(countries_2018, file_w='results/winners_fixed_seed.txt', file_s='results/semis_fixed_seed.txt')
dw_random = analyze_MC_samples(countries_2018, file_w='results/winners_rand_seed.txt',  file_s='results/semis_rand_seed.txt')

generate_plots(dw_fixed, dw_random)


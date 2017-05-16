#!/usr/bin/env python
# Martin Kersner, 2016/01/13
# modified by Holger Roth, 2017

from __future__ import print_function
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cPickle as pickle
import os
import glob
from scipy.signal import savgol_filter, medfilt
from matplotlib.ticker import ScalarFormatter

from utils import strstr

def main():
  #log_files = process_arguments(sys.argv)
  
  # get newest log
  log_dir = 'log'
  log_files = [max(glob.iglob(os.path.join(log_dir, '*.log')), key=os.path.getctime)]

  print(log_files)

  XLIM = []
  YLIM = []
  #XLIM = (0, 1000)
  #YLIM = (0.06, 0.5)

  LOGSCALE = False
  #LOGSCALE = True
  
  REMOVE_ZERO_LOSS = True

  train_iteration = []
  train_loss      = []
  test_iteration = []
  test_loss      = []  
  base_test_iter  = 0
  base_train_iter = 0 
  curr_train_loss = []
  curr_test_loss = []  

  for log_file in log_files:
    with open(log_file, 'rb') as f:
      if len(train_iteration) != 0:
        base_train_iter = train_iteration[-1]
        base_test_iter = test_iteration[-1]

      line_count = 0
      for line in f:
        line_count += 1  
        #print('{}: {}'.format(line_count, line))
        if strstr(line, 'Iteration') and strstr(line, ', loss'):
          matched = match_iteration(line)
          train_iteration.append(int(matched.group(1))+base_train_iter)
          TEST_LOSS = False
          TRAIN_LOSS = True
        elif strstr(line, 'Iteration') and strstr(line, 'loss'):
          TEST_LOSS = True  
          TRAIN_LOSS = False
          matched = match_test_iteration(line)
          test_iteration.append(int(matched.group(1))+base_test_iter)
        else:
          TEST_LOSS = False
          TRAIN_LOSS = False

        # TRAIN LOSS
        if TRAIN_LOSS:
            matched = match_train_loss(line)
            if matched:
              curr_train_loss.append( float(matched.group(1)) )
              #print('#{} train loss: {}'.format(k_train,float(matched.group(2))))
              train_loss.append( np.sum(curr_train_loss) )    
              print('Iter {}: total train loss: {}'.format(train_iteration[-1],np.sum(curr_train_loss)))              
              curr_train_loss = []

        # TEST LOSS
        if TEST_LOSS:
            matched = match_test_loss(line)
            if matched:
              curr_test_loss.append( float(matched.group(1)) )
              #print('#{} test loss: {}'.format(k_test,float(matched.group(2))))
              test_loss.append( np.sum(curr_test_loss) )    
              print('Iter {}: total test loss: {}'.format(test_iteration[-1],np.sum(curr_test_loss)))                  
              curr_test_loss = []
                            
  log_base = os.path.splitext(os.path.basename(log_files[0]))[0]
 
  result = {'TRAIN': (train_iteration,train_loss)}
  pickle.dump( result , open( log_base+'.pkl', "wb" ) )

  print('read {} lines'.format(line_count))
  print("TRAIN", np.shape(train_iteration), np.shape(train_loss))
  print("TEST", np.shape(test_iteration), np.shape(test_loss))

  if REMOVE_ZERO_LOSS:
      # convert to numpy
      Ntrain0 = len(train_loss)
      train_loss = np.array(train_loss)
      train_iteration = np.array(train_iteration)
      
      idx = train_loss>0.0      
      train_iteration = train_iteration[idx]
      train_loss = train_loss[idx]
      print('Removed {} zeros: Size changed from {} to {}'.format(np.sum(idx),Ntrain0,np.size(train_loss)))

  if len(train_loss) < len(train_iteration):
      Ntrain = len(train_loss)
  else:
      Ntrain = len(train_iteration)
  if len(test_loss) < len(test_iteration):
      Ntest = len(test_loss)
  else:
      Ntest = len(test_iteration)
  
  if Ntrain>0:
      print("Best TRAIN performance at index:")
      min_train_loss = np.min(train_loss)
      best_idx = np.where(train_loss==min_train_loss)[0][0]
      print(best_idx)
      print("{}: iteration {}, loss {}".format(log_base,train_iteration[best_idx],min_train_loss))

  if Ntest>0:
      print("Best TEST performance at index:")
      min_test_loss = np.min(test_loss)
      best_idx = np.where(test_loss==min_test_loss)[0][0]
      print(best_idx)
      print("{}: iteration {}, loss {}".format(log_base,test_iteration[best_idx],min_test_loss))  

  # Smoothing
  window = int(0.1*float(Ntrain))
  window = -int(0.1*float(Ntrain)) # median
  if abs(window) > 100:
      window = 100*abs(window)/window # keep sign
  if window % 2 == 0: # make uneven
      window = window + 1

  testwindow = int(0.1*float(Ntest))
  testwindow = -int(0.1*float(Ntest)) # median
  if abs(testwindow) > 100:
      testwindow = 100*abs(testwindow)/testwindow # keep sign
  if testwindow % 2 == 0: # make uneven
      testwindow = testwindow + 1
      
  if Ntrain>0 and window < Ntrain:
    print('smoothing {} data points with window of {}'.format(Ntrain, window))           
    if window > 0 and window < Ntrain and window>3:
      train_loss_smooth = savgol_filter(train_loss, abs(window), 3) # window size, polynomial order
    else:  
      train_loss_smooth = medfilt(train_loss, abs(window)) # window size 51
  else:
    train_loss_smooth = []
  if Ntest>0 and window < Ntest:
    print('smoothing {} data points with window of {}'.format(Ntest, testwindow)) 
    if window > 0 and window>3:
      test_loss_smooth = savgol_filter(test_loss, abs(testwindow), 3) # window size, polynomial order
    else:
      test_loss_smooth = medfilt(test_loss, abs(testwindow)) # window size 51
  else:
    test_loss_smooth = []

  ##print("Best smoothed TRAIN performance at index:")
  ##min_train_loss = np.min(train_loss_smooth[window:Ntrain-abs(window/2)])
  ##best_idx = np.where(train_loss_smooth[window:Ntrain-abs(window/2)]==min_train_loss)[0][0] + -abs(window/2)
  ##print(best_idx)
  ##print("{}: iteration {}, loss {}".format(log_base,train_iteration[best_idx],min_train_loss))

  ##print("Best smoothed TEST performance at index:")
  ##min_test_loss = np.min(test_loss_smooth[window:Ntest-abs(window/2)])
  ##best_idx = np.where(test_loss_smooth[window:Ntest-abs(window/2)]==min_test_loss)[0][0] + -abs(window/2)
  ##print(best_idx)
  ##print("{}: iteration {}, loss {}".format(log_base,test_iteration[best_idx],min_test_loss))
      
  # Visualization
  plot_loss(train_iteration,train_loss,test_iteration,test_loss,LOGSCALE,XLIM,YLIM,log_base)
  plot_loss(train_iteration,train_loss_smooth,test_iteration,test_loss_smooth,LOGSCALE,XLIM,YLIM,log_base+'_smooth')

def plot_loss(train_iteration,train_loss,test_iteration,test_loss,LOGSCALE=False,XLIM=[],YLIM=[],log_base=[]):
  iter_range = np.abs( np.max(train_iteration)  - np.min(train_iteration) )
  plt.figure()
  plt.hold(True)
  if LOGSCALE:
      plt.semilogy(train_iteration, train_loss, '', label='Train loss')
      plt.semilogy(test_iteration, test_loss, 'r', label='Test loss')
  else:
      plt.plot(train_iteration, train_loss, '', label='Train loss')
      plt.plot(test_iteration, test_loss, 'r', label='Test loss')      
  if XLIM:
      plt.xlim(XLIM)
  if YLIM:
      plt.ylim(YLIM)      
  plt.legend()
  plt.ylabel('Loss')
  plt.xlabel('Number of iterations')
  plt.grid(True)
  plt.tight_layout() 
  ax = plt.axes()
  train_loss_range=np.abs(np.max(train_loss)-np.min(train_loss))
  print('train_loss range smooth: {}'.format(train_loss_range))
  ax.xaxis.set_major_locator(ticker.MultipleLocator(int(0.5*float(iter_range))))
  ax.xaxis.set_minor_locator(ticker.MultipleLocator(int(0.1*float(iter_range))))     
  if LOGSCALE:
      ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[0.1, 1.0], numdecs=4, numticks=50))
      ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[1]))      
  else:
      ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05*train_loss_range))
      ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01*train_loss_range))    
  ax.yaxis.set_major_formatter(ScalarFormatter())            

  plt.title(log_base)
  plt.draw()
  if log_base:
      plt.savefig(log_base+'.png')    

def match_iteration(line):
  line = line[0:line.find(',')]
  return re.search(r'Iteration (.*)', line)

def match_test_iteration(line):
  line = line[0:line.find(' loss')]
  return re.search(r'Iteration (.*)', line)
  
def match_loss(line):
  line = line[0:line.find(' loss')]
  return re.search(r'loss = (.*)', line)

def match_train_loss(line):
  line = line[line.find(' loss')::]
  return re.search(r'loss = (.*)', line)  
  
def match_test_loss(line):
  line = line[line.find('loss')::]
  return re.search(r'loss (.*)', line)    

def process_arguments(argv):
  print(argv)
  if len(argv) < 2:
    help()

  log_files = argv[1:]
  return log_files

def help():
  print('Usage: python loss_from_log.py [LOG_FILE]+\n'
        'LOG_FILE is text file containing log produced by caffe.'
        'At least one LOG_FILE has to be specified.'
        'Files has to be given in correct order (the oldest logs as the first ones).'
        , file=sys.stderr)

  exit()

if __name__ == '__main__':
  main()

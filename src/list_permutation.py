import numpy as np
import itertools
import random
from scipy.spatial.distance import cdist

class list_permutation():
  def __init__(self, classes):
    #get no of classes from the jigsaw puzzle class.
    self.classes = classes
  
  def get_permutation(self):
    p_list = []
    P_hat = np.array(list(itertools.permutations(list(range(9)), 9)))
    n = P_hat.shape[0]
    for i in range(self.classes):
      if i==0:
        j = np.random.randint(n)
        P = np.array(P_hat[j]).reshape([1,-1])
        #n = n-1
      else:
        #j = np.random.randint(n)
        P = np.concatenate([P,P_hat[j].reshape([1,-1])],axis=0)
      P_hat = np.delete(P_hat,j,axis=0)
      D = cdist(P,P_hat, metric='hamming').mean(axis=0).flatten()
      j=D.argmax()
    np.save('/netscratch/mundra/svhnpermutations/permutations_%s'%(self.classes),P)
      #m = int(D.shape[0]/2)
      #S = D.argsort()
      #j = S[np.random.randint(m-10,m+10)]
    #return P

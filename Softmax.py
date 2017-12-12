import numpy as np
import pandas as pd
import math
import random
from operator import itemgetter
import datetime
from sklearn.cross_validation import train_test_split

class Softmax:
    def __init__(self, batch_size=50, epochs=1000, learning_rate=1e-2, reg_strength=1e-5, weight_update='adam'):
        self.W = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.weight_update = weight_update

    def train(self, X, y):
        n_features = X.shape[1]
        n_classes = y.max() + 1
        self.W = np.random.randn(n_features, n_classes) / np.sqrt(n_features/2)
        config = {'reg_strength': self.reg_strength, 'batch_size': self.batch_size,
                'learning_rate': self.learning_rate, 'eps': 1e-8, 'decay_rate': 0.99,
                'momentum': 0.9, 'cache': None, 'beta_1': 0.9, 'beta_2':0.999,
                'velocity': np.zeros(self.W.shape)}
        c = globals()['Softmax']
        for epoch in range(self.epochs):
            loss, config = getattr(c, self.weight_update)(self, X, y, config)

    def predict(self, X):
        return np.argmax(X.dot(self.W), 1)

    def loss(self, X, y, W, b, reg_strength):
        sample_size = X.shape[0]
        predictions = X.dot(W) + b

        predictions -= predictions.max(axis=1).reshape([-1, 1])

        # Run predictions through softmax
        softmax = math.e**predictions
        softmax /= softmax.sum(axis=1).reshape([-1, 1])

        # Cross entropy loss
        loss = -np.log(softmax[np.arange(len(softmax)), y]).sum() 
        loss /= sample_size
        loss += 0.5 * reg_strength * (W**2).sum()

        softmax[np.arange(len(softmax)), y] -= 1
        dW = (X.T.dot(softmax) / sample_size) + (reg_strength * W)
        return loss, dW

    def sgd(self, X, y, config):
        items = itemgetter('learning_rate', 'batch_size', 'reg_strength')(config)
        learning_rate, batch_size, reg_strength = items

        loss, dW = self._calculate_gradient(X, y, batch_size, self.W, 0, reg_strength)
 
        self.W -= learning_rate * dW
        return loss, config

    def sgd_with_momentum(self, X, y, config):
        items = itemgetter('learning_rate', 'batch_size', 'reg_strength', 'momentum')(config)
        learning_rate, batch_size, reg_strength, momentum = items

        loss, dW = self._calculate_gradient(X, y, batch_size, self.W, 0, reg_strength)

        config['velocity'] = momentum*config['velocity'] - learning_rate*dW
        self.W += config['velocity']
        return loss, config

    def rms_prop(self, X, y, config):
        items = itemgetter('learning_rate', 'batch_size', 'reg_strength', 'decay_rate', 'eps', 'cache')(config)
        learning_rate, batch_size, reg_strength, decay_rate, eps, cache = items

        loss, dW = self._calculate_gradient(X, y, batch_size, self.W, 0, reg_strength)

        cache = np.zeros(dW.shape) if cache == None else cache
        cache = decay_rate * cache + (1-decay_rate) * dW**2
        config['cache'] = cache

        self.W -= learning_rate * dW / (np.sqrt(cache) + eps)
        return loss, config

    def adam(self, X, y, config):
        items = itemgetter('learning_rate', 'batch_size', 'reg_strength', 'eps', 'beta_1', 'beta_2')(config)
        learning_rate, batch_size, reg_strength, eps, beta_1, beta_2 = items
        config.setdefault('t', 0)
        config.setdefault('m', np.zeros(self.W.shape))
        config.setdefault('v', np.zeros(self.W.shape))

        loss, dW = self._calculate_gradient(X, y, batch_size, self.W, 0, reg_strength)

        config['t'] += 1
        config['m'] = config['m']*beta_1 + (1-beta_1)*dW
        config['v'] = config['v']*beta_2 + (1-beta_2)*dW**2
        m = config['m']/(1-beta_1**config['t'])
        v = config['v']/(1-beta_2**config['t'])
        self.W -= learning_rate*m/(np.sqrt(v)+eps)
        return loss, config

    def _calculate_gradient(self, X, y, batch_size, w, b, reg_strength):
        random_indices = random.sample(range(X.shape[0]), batch_size)
        X_batch = X[random_indices]
        y_batch = y[random_indices]
        return self.loss(X_batch, y_batch, w, b, reg_strength)


dtype={'is_booking':bool,
        'srch_ci' : np.str_,
        'srch_co' : np.str_,
        'srch_adults_cnt' : np.int32,
        'srch_children_cnt' : np.int32,
        'srch_rm_cnt' : np.int32,
        'srch_destination_id':np.str_,
        'user_location_country' : np.str_,
        'user_location_region' : np.str_,
        'user_location_city' : np.str_,
        'hotel_cluster' : np.str_,
        'orig_destination_distance':np.float64,
        'date_time':np.str_,
        'hotel_market':np.str_
      }

import pandas as pd
df0=pd.read_csv("train.csv",dtype=dtype, usecols=dtype, parse_dates=['date_time'] ,sep=',').dropna()

df0 = df0[(df0['hotel_cluster'] == '91')  | (df0['hotel_cluster'] == '41') | (df0['hotel_cluster'] == '48') | (df0['hotel_cluster'] == '25') | 
           (df0['hotel_cluster'] == '33') | (df0['hotel_cluster'] == '65') | (df0['hotel_cluster'] == '95') | (df0['hotel_cluster'] == '18') |
          (df0['hotel_cluster'] == '21')]

df0['year']=df0['date_time'].dt.year
train = df0.query('is_booking==True')

train['srch_ci']=pd.to_datetime(train['srch_ci'],infer_datetime_format = True,errors='coerce')
train['srch_co']=pd.to_datetime(train['srch_co'],infer_datetime_format = True,errors='coerce')

train['month']= train['date_time'].dt.month
train['plan_time'] = ((train['srch_ci']-train['date_time'])/np.timedelta64(1,'D')).astype(float)
train['hotel_nights']=((train['srch_co']-train['srch_ci'])/np.timedelta64(1,'D')).astype(float)

m=train.orig_destination_distance.mean()
train['orig_destination_distance']=train.orig_destination_distance.fillna(m)
train.fillna(-1,inplace=True)
lst_drop=['date_time','srch_ci','srch_co']
train.drop(lst_drop,axis=1,inplace=True)

y=train['hotel_cluster']
X=train.drop(['hotel_cluster','is_booking','year','plan_time','month','user_location_city','hotel_market'],axis=1)

X["user_location_country"]=X["user_location_country"].astype(int)
X["user_location_region"]=X["user_location_region"].astype(int)
X["srch_destination_id"]=X["srch_destination_id"].astype(int)

X1=X.as_matrix()
y1=np.array(y)

X_train=X1[0:280000]
Y_train=y1[0:280000]
X_test=X1[280000:]
Y_test=y1[280000:]


weight_update = 'sgd'
sm = Softmax(weight_update=weight_update)
sm.train(X_train, Y_train)
pred = sm.predict(X_test)
print (np.mean(np.equal(Y_test, pred)))


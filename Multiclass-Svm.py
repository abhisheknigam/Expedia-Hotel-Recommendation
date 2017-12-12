import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder

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
df0=pd.read_csv("train.csv",dtype=dtype, names=dtype, parse_dates=['date_time'] ,sep=',').dropna()

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


def get_projection_weight(v, z=1):
    """
    Projection calculation:
        w^* = argmin_w 0.5 ||w-v||^2 s.t. \sum_i w_i = z, w_i >= 0
    """
    
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


class CustomSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1, max_iterations=50, tolerance=0.05,
                 random_state=None, verbose=0):
        self.C = C
        self.max_iterations = max_iterations
        self.tolerance = tolerance,
        self.random_state = random_state
        self.verbose = verbose

    def _calc_gradient(self, X, y, i):
        # Partial gradient for the ith sample.
        g = np.dot(X[i], self.coef_.T) + 1
        g[y[i]] -= 1
        return g

    def _get_violation_score(self, g, y, i):
        smallest = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.dual_coef_[k, i] >= self.C:
                continue
            elif k != y[i] and self.dual_coef_[k, i] >= 0:
                continue

            smallest = min(smallest, g[k])

        return g.max() - smallest

    def _solve_subproblem(self, g, y, norms, i):
        # Prepare inputs to the projection.
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.dual_coef_[:, i]) + g / norms[i]
        z = self.C * norms[i]
        beta = get_projection_weight(beta_hat, z)

        return Ci - self.dual_coef_[:, i] - beta / norms[i]

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Normalizing labels.
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)

        # Coefficients initialization.
        n_classes = len(self._label_encoder.classes_)
        self.dual_coef_ = np.zeros((n_classes, n_samples), dtype=np.float64)
        self.coef_ = np.zeros((n_classes, n_features))
        
        norms = np.sqrt(np.sum(X ** 2, axis=1))

        # Shuffling indexes.
        rs = check_random_state(self.random_state)
        ind = np.arange(n_samples)
        rs.shuffle(ind)

        violation_init = None
        for it in range(self.max_iterations):
            violation_sum = 0

            for ii in range(n_samples):
                i = ind[ii]
                if norms[i] == 0:
                    continue

                g = self._calc_gradient(X, y, i)
                v = self._get_violation_score(g, y, i)
                violation_sum += v

                if v < 1e-12:
                    continue

                # Solve subproblem for ith sample.
                delta = self._solve_subproblem(g, y, norms, i)
                self.coef_ += (delta * X[i][:, np.newaxis]).T
                self.dual_coef_[:, i] += delta

            if it == 0:
                violation_init = violation_sum

            vratio = violation_sum / violation_init

            if self.verbose >= 1:
                print("iter", it + 1, "violation", vratio)

            if vratio < self.tolerance:
                if self.verbose >= 1:
                    print("Converged")
                break

        return self

    def predict(self, X):
        decision = np.dot(X, self.coef_.T)
        pred = decision.argmax(axis=1)
        return self._label_encoder.inverse_transform(pred)


clf = CustomSVM(C=1.0, tolerance=0.01, max_iterations=100, random_state=0, verbose=1)
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)


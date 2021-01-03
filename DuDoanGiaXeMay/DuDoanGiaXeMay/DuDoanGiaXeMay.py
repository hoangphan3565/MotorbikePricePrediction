# In[0]: IMPORTS 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix   

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import joblib

run_new_evaluate = False
run_new_search = False
new_run_ensemble = False
let_plot=True

# In[2] Get data
raw_data = pd.read_csv(r'data\Data_XeMay.csv')

# In[3]: Discover the data to gain insights
# 3.1 Quick view of the data
print('\n____________________________________ Dataset info ____________________________________')
print(raw_data.info())   
# -> xem  sơ bộ dữ liệu: có tất cả 5 feature, non-null count là số lượng ô ko bị trống dữ liệu, 
# dtype là kiểu của dữ liệu
# các feature tương đối đầy đủ dữ liệu trừ DUNG TÍCH - CC trống 2000 
print('\n____________________________________ Some first data examples ____________________________________')
print(raw_data.head(6)) 
# -> xem 6 dòng dữ liệu đầu tiên

print('\n____________________________________ Counts on a feature ____________________________________')
print(raw_data['HÃNG'].value_counts()) # 17 cate
print(raw_data['DÒNG'].value_counts()) # bị khuyết -> 118
print(raw_data['KIỂU XE'].value_counts()) # 4
print(raw_data['TÌNH TRẠNG'].value_counts()) # 2
# dự sẽ có 17 + 118 + 4 + 2 = 141 onehot vector feature

print('\n____________________________________ Statistics of numeric features ____________________________________')
print(raw_data.describe())   
#                STT  GIÁ - TRIỆU ĐỒNG        NĂM_SX 
# count  10000.00000      10000.000000  10000.000000
# mean    5000.50000         36.367398   2012.582800
# std     2886.89568         56.647178      6.503612
# min        1.00000          1.200000   1979.000000
# 25%     2500.75000         11.500000   2010.000000
# 50%     5000.50000         23.000000   2014.000000
# 75%     7500.25000         38.500000   2018.000000
# max    10000.00000        699.000000   2020.000000

# xem xét thấy giá xe thấp nhất và cao nhất ko có bất thường

 
print('\n____________________________________ Get specific rows and cols ____________________________________')     
print(raw_data.iloc[[0,5,20], [5, 3]] ) # Refer using column ID



# 3.2 Scatter plot b/w 2 features
if let_plot:
    raw_data.plot(kind="scatter", y="GIÁ - TRIỆU ĐỒNG", x="DUNG TÍCH - CC", alpha=0.2)
    plt.savefig('figures/scatter_1_feat.png', format='png', dpi=300)
    plt.show()  
    # -> dung tích càng lớn giá xe càng cao

if let_plot:
    raw_data.plot(kind="scatter", y="GIÁ - TRIỆU ĐỒNG", x="NĂM_SX", alpha=0.2)
    plt.savefig('figures/scatter_2_feat.png', format='png', dpi=300)
    plt.show()  
    # xu hướng xe được sản xuất càng gần đây giá càng cao, 
    # có một vài ngoại lệ đó là xe trong khoảng trước và sau 1980 có giá tương đối cao, 
    # có thể do đây là xe cổ 

if let_plot:
    raw_data.plot(kind="scatter", y="GIÁ - TRIỆU ĐỒNG", x="KIỂU XE", alpha=0.2)
    plt.savefig('figures/scatter_3_feat.png', format='png', dpi=300)
    plt.show()  
    # xe Tay côn/mô tô có một số sample có giá rất cao
 
if let_plot:
    raw_data.plot(kind="scatter", y="GIÁ - TRIỆU ĐỒNG", x="TÌNH TRẠNG", alpha=0.2)
    plt.savefig('figures/scatter_4_feat.png', format='png', dpi=300)
    plt.show()  

# 3.3 Scatter plot b/w every pair of features
if let_plot:
    features_to_plot = ["GIÁ - TRIỆU ĐỒNG", "NĂM_SX"]
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.savefig('figures/scatter_mat_all_feat.png', format='png', dpi=300)
    plt.show()

# 3.4 Plot histogram of 1 feature
if let_plot:
    features_to_plot = ['NĂM_SX']
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.show()

    features_to_plot2 = ['GIÁ - TRIỆU ĐỒNG']
    scatter_matrix(raw_data[features_to_plot2], figsize=(12, 8))
    plt.savefig('figures/scatter_mat_giatien_feat.png', format='png', dpi=300)
    plt.show()
# 3.5 Plot histogram of numeric features
if let_plot:
    raw_data.hist(figsize=(10,5)) #bins: no. of intervals
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.tight_layout()
    plt.savefig('figures/hist_raw_data.png', format='png', dpi=300) # must save before show()
    plt.show()

# 3.6 Compute correlations b/w features
print('3.6 Compute correlations b/w features')
corr_matrix = raw_data.corr()
print(corr_matrix) # print correlation matrix
print(corr_matrix["GIÁ - TRIỆU ĐỒNG"].sort_values(ascending=False))
# Feature DUNG TÍCH - CC      0.812330
#         NĂM_SX              0.226656


## 3.7 Try combining features - thử kết hợp các thuộc tính lại với nhau và đo mức độ tương quan
#print('3.7 Try combining features')
#raw_data["DUNG TÍCH + NĂM"] = raw_data["DUNG TÍCH - CC"] + raw_data["NĂM_SX"]    
#raw_data["DUNG TÍCH X NĂM"] = raw_data["DUNG TÍCH - CC"] * raw_data["NĂM_SX"]        
#corr_matrix = raw_data.corr()
#print(corr_matrix["GIÁ - TRIỆU ĐỒNG"].sort_values(ascending=False)) # print correlation b/w a feature and other features
#raw_data.drop(columns = ["DUNG TÍCH + NĂM","DUNG TÍCH X NĂM"], inplace=True) # remove experiment columns

# Vì độ tương quan sau khi thử kết hợp 2 feature này ko thay đổi nhiều, và dataset hiện tại không đủ thêm feature để thử kết hợp
# do đó trong task này không thử kết hợp tạo thêm feature mới trong quá trình training.


# In[04]: PREPARE THE DATA 
# 4.1 Remove unused features - xóa dững thuộc tính không cần thiết
raw_data.drop(columns = ["STT"], inplace=True) 

# 4.2 Split training-test set and NEVER touch test set until test phase # Create new feature "KHOẢNG GIÁ": the distribution we want to remain
raw_data["KHOẢNG GIÁ"] = pd.cut(raw_data["GIÁ - TRIỆU ĐỒNG"],
                                bins=[0, 5, 10, 15, 20, 25,35,50,100,200,300, np.inf],
                                #labels=["<5 triệu", "5-10 triệu", "15-20 triệu",...]
                                labels=[5, 10, 15, 20, 25,35,50,100,200,300,700]) # use numeric labels to plot histogram
# chia dữ liệu trên cột giá thành từng đoạn và đếm vào cột khoảng giá

# Create training and test set - dùng thư viện của sklearn để chia test set để tránh Stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit  
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42) # n_splits: no. of re-shuffling & splitting = no. of train-test sets 
                                                                              # (if you want to run the algorithm n_splits times with different train-test set)
for train_index, test_index in splitter.split(raw_data, raw_data["KHOẢNG GIÁ"]): # Feature "KHOẢNG GIÁ" must NOT contain NaN
    train_set = raw_data.loc[train_index]
    test_set = raw_data.loc[test_index]      
# n_splits = 1 vì chỉ chia 1 lần, vòng for để tổng quát với các bước ở sau. trong th này vòng for chỉ chạy 1 lần

# See if it worked as expected
if 0:
    raw_data["KHOẢNG GIÁ"].hist(bins=6, figsize=(5,5)); #plt.show();
    train_set["KHOẢNG GIÁ"].hist(bins=6, figsize=(5,5)); plt.show()

# Remove the new feature
print(train_set.info())
for _set_ in (train_set, test_set):
    _set_.drop(columns="KHOẢNG GIÁ", inplace=True) 
print(train_set.info())
print(test_set.info())
print('\n____________________________________ Split training an test set ____________________________________')     
print(len(train_set), "train +", len(test_set), "test examples")
print(train_set.head(4))



# 4.3 tách lalels ra khỏi dữ liệu
train_set_labels = train_set["GIÁ - TRIỆU ĐỒNG"].copy()
train_set = train_set.drop(columns = "GIÁ - TRIỆU ĐỒNG") 
test_set_labels = test_set["GIÁ - TRIỆU ĐỒNG"].copy()
test_set = test_set.drop(columns = "GIÁ - TRIỆU ĐỒNG") 

# 4.4 ịnh nghĩa 1 luồng xử lý liên tục để xử lý dữ liệu, 
# mục đích là để chuyển tất cả các feature dạng chữ thành dạng số để thuật toán có thể học được
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values  
num_feat_names = ['DUNG TÍCH - CC','NĂM_SX']
cat_feat_names = ['HÃNG','DÒNG','KIỂU XE','TÌNH TRẠNG']

## 4.4.2 Pipeline for categorical features
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)), # chọn dữ liệu trong cat_feat_names
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place - điền khuyết
    ('cat_encoder', OneHotEncoder()) # convert categorical data into one-hot vectors - chuyển thành dạng one-hot vector
    ])   


# 4.4.4 pineline để xử lý các feature có dạng số
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)), # chọn từ num_feat_names
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)), # copy=False: imputation will be done in-place - điền khuyết theo dạng median
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) # Scale features to zero mean and unit variance # co dãn dữ liệu để thuật toán chạy nhanh và chính xác hơn
    ]) 
# 4.4.5 kết hợp 2 pineline đã viết lại thành 1 pineline 
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  

# 4.5 chạy pineline để xử lý training data
processed_train_set_val = full_pipeline.fit_transform(train_set)



print('\n____________________________________ Processed feature values ____________________________________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)
# (9000, 143)
print('We have %d numeric feature 141 cols of onehotvector for categorical features.' %(len(num_feat_names)))
# Trên training set có 9000 sample 143 feature gồm:
## 2 numeric feature 
## 141 one-hot vector được chuyển từ dữ liệu dạng categorical

# In[5]: TRAIN AND EVALUATE MODELS huấn luyện và tìm ra model tốt nhất

# Tạo hàm lưu và load các model
def store_model(model, model_name = ""):
    # NOTE: sklearn.joblib faster than pickle of Python
    # INFO: can store only ONE object in a file
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'saved_objects/' + model_name + '_model.pkl')
def load_model(model_name):
    # Load objects into memory
    #del model
    model = joblib.load('saved_objects/' + model_name + '_model.pkl')
    #print(model)
    return model
# Hàm tính điểm  r2score và rmse
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse  


#    Input data:
#          HÃNG       DÒNG  NĂM_SX  DUNG TÍCH - CC       KIỂU XE TÌNH TRẠNG
#         Honda       Wave  2018.0           110.0         Xe số         Cũ
#         Honda     Winner  2018.0           150.0  Tay côn/Moto         Cũ
#         Honda  Air Blade  2012.0           125.0        Tay ga         Cũ
#       Piaggio      Vespa  2012.0           125.0        Tay ga         Cũ
#        Yamaha    Exciter  2012.0           125.0  Tay côn/Moto         Cũ
#        Yamaha    Exciter  2013.0           125.0  Tay côn/Moto         Cũ
#         Honda      Sonic  2018.0           150.0  Tay côn/Moto         Cũ
#         Honda  Air Blade  2010.0           110.0        Tay ga         Cũ
#         Honda  Air Blade  2008.0           110.0        Tay ga         Cũ

if 0:
    # 5.1 Thử với model LinearRegression
    # 5.1.1 Training: learn a linear regression hypothesis using training data 
    model = LinearRegression()
    model.fit(processed_train_set_val, train_set_labels)
    print('\n____________________________________ LinearRegression ____________________________________')
    print('Learned parameters: ', model.coef_)

    # 5.1.2 Tính R2 score và  RMSE
    r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
    print('R2 score (on training data, best=1):', r2score)
    print("Root Mean Square Error: ", rmse.round(decimals=1))
    # R2 score (on training data, best=1): 0.9384642356070718
    # Root Mean Square Error:  14.5


    # 5.1.3 so sánh dữ liệu thực và dữ liệu được đoán bởi LinearRegression Model
    print("Input data: \n", train_set.iloc[0:9])
    print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
    print("Labels:      ", list(train_set_labels[0:9]))
    #Predictions:  [16.6  27.7  22.7  29.3  23.7  24.6  46.1  16.   14.2]
    #Labels:       [15.3, 27.9, 23.0, 15.8, 20.0, 17.8, 43.8, 16.0, 15.0]
 
if 0:
    # 5.2 Try DecisionTreeRegressor model
    # Training
    model = DecisionTreeRegressor()
    model.fit(processed_train_set_val, train_set_labels)
    # Compute R2 score and root mean squared error
    print('\n____________________________________ DecisionTreeRegressor ____________________________________')
    r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
    print('R2 score (on training data, best=1):', r2score)
    print("Root Mean Square Error: ", rmse.round(decimals=1))
    # R2 score (on training data, best=1): 0.9875223214633033
    # Root Mean Square Error:  6.5

    store_model(model)
    # Predict labels for some training instances
    print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
    print("Labels:      ", list(train_set_labels[0:9]))
    # Predictions:  [14.1  26.9  19.7  18.7  19.1  19.5  42.7  18.3  14.1]
    # Labels:       [15.3, 27.9, 23.0, 15.8, 20.0, 17.8, 43.8, 16.0, 15.0]
    
if 0:
    # 5.3 Try RandomForestRegressor model
    # Training (NOTE: Mất nhiều tgian nếu tập train lớn)
    model = RandomForestRegressor(n_estimators=100, random_state=42) # n_estimators: số lượng cây
    model.fit(processed_train_set_val, train_set_labels)
    # Compute R2 score and root mean squared error
    print('\n____________________________________ RandomForestRegressor ____________________________________')
    r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
    print('R2 score (on training data, best=1):', r2score)
    print("Root Mean Square Error: ", rmse.round(decimals=1))
    # R2 score (on training data, best=1): 0.9863534309591574
    # Root Mean Square Error:  6.8

    store_model(model)      
    # Predict labels for some training instances
    print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
    print("Labels:      ", list(train_set_labels[0:9]))
    # Labels:       [15.3, 27.9, 23.0, 15.8, 20.0, 17.8, 43.8, 16.0, 15.0]
    # Predictions:  [14.1  26.9  19.7  18.6  19.   19.5  42.8  18.3  14. ]

if 0:
    # 5.5 Try Support Vector Machine Regressor model
    model = SVR(kernel='rbf',epsilon=2,C=10000)
    model.fit(processed_train_set_val, train_set_labels)
    # Compute R2 score and root mean squared error
    print('\n____________________________________ SupportVectorMachineRegressor ____________________________________')
    r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
    print('R2 score (on training data, best=1):', r2score)
    print("Root Mean Square Error: ", rmse.round(decimals=1))
    # R2 score (on training data, best=1): 0.978840172678805
    # Root Mean Square Error:  8.5

    store_model(model)
    # Predict labels for some training instances
    print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
    print("Labels:      ", list(train_set_labels[0:9]))
    # Predictions:  [13.8  27.5  20.4  18.9  20.5  19.6  44.3  16.5  14.7]
    # Labels:       [15.3, 27.9, 23.0, 15.8, 20.0, 17.8, 43.8, 16.0, 15.0]

if 0:
    # 5.6 Try Voting Ensemble
    r1 = RandomForestRegressor(n_estimators=100, random_state=42)
    r2 = SVR(kernel='rbf',epsilon=2,C=10000)
    model = VotingRegressor([('rf', r1),('svr',r2)])
    model.fit(processed_train_set_val, train_set_labels)
    print('\n____________________________________ EnsembleRegressor ____________________________________')
    r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
    print('R2 score (on training data, best=1):', r2score)
    print("Root Mean Square Error: ", rmse.round(decimals=1))
    # R2 score (on training data, best=1): 0.9845055782541611
    # Root Mean Square Error:  7.3

    store_model(model)
    # Predict labels for some training instances
    print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
    print("Labels:      ", list(train_set_labels[0:9]))
    # Predictions:  [14.   27.2  20.   18.8  19.7  19.6  43.5  17.4  14.4]
    # Labels:       [15.3, 27.9, 23.0, 15.8, 20.0, 17.8, 43.8, 16.0, 15.0]

if 0: 
    # 5.7 Try Extra-tree
    model = ExtraTreesRegressor(n_estimators=200, bootstrap=True, random_state=42)
    model.fit(processed_train_set_val, train_set_labels)
    print('\n____________________________________ ExtraTreesRegressor ____________________________________')
    r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
    print('R2 score (on training data, best=1):', r2score)
    print("Root Mean Square Error: ", rmse.round(decimals=1))
    # R2 score (on training data, best=1): 0.9867715955998396
    # Root Mean Square Error:  6.7
    
    store_model(model)
    # Predict labels for some training instances
    print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
    print("Labels:      ", list(train_set_labels[0:9]))
    # Predictions:  [14.1  26.8  19.7  18.7  19.   19.5  42.7  18.3  14. ]
    # Labels:       [15.3, 27.9, 23.0, 15.8, 20.0, 17.8, 43.8, 16.0, 15.0]

if 0: 
    # 5.8 Try AdaBoost
    model = AdaBoostRegressor(
            ExtraTreesRegressor(n_estimators=200, bootstrap=True, random_state=42), 
            n_estimators=10, random_state=42)
    model.fit(processed_train_set_val, train_set_labels)
    print('\n____________________________________ AdaBoostRegressor ____________________________________')
    r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
    print('R2 score (on training data, best=1):', r2score)
    print("Root Mean Square Error: ", rmse.round(decimals=1))
    # R2 score (on training data, best=1): 0.9532637759163574
    # Root Mean Square Error:  12.6

    store_model(model)
    # Predict labels for some training instances
    print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
    print("Labels:      ", list(train_set_labels[0:9]))
    # Predictions:  [14.2  28.   20.9  18.5  19.5  19.8  44.9  16.4  15. ]
    # Labels:       [15.3, 27.9, 23.0, 15.8, 20.0, 17.8, 43.8, 16.0, 15.0]


# 5.6 Đánh giá các model bằng K-fold cross validation
print('\n____________________________________ K-fold cross validation ____________________________________')

if run_new_evaluate:
    # Evaluate LinearRegression
    model_name = "LinearRegression" 
    model = LinearRegression()             
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))
    # => LinearRegression rmse:  [16.7 12.9 11.6 14.  18.6]


    # => DecisionTreeRegressor rmse:  [13.1  8.4 10.8 14.  14.1]


    # Evaluate RandomForestRegressor
    model_name = "RandomForestRegressor" 
    model = RandomForestRegressor(n_estimators=100, random_state=42) # n_estimators: số lượng cây
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    # => RandomForestRegressor rmse:  [ 9.6  7.2  8.1 12.7 13.8]


    # Evaluate SVR
    model_name = "SupportVectorMachineRegressor" 
    model = SVR(kernel='rbf',epsilon=2,C=10000)             
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("SupportVectorMachineRegressor rmse: ", rmse_scores.round(decimals=1))
    # => [ 8.   8.1  7.  10.  14.4]


else:
    # Load rmse
    model_name = "LinearRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "DecisionTreeRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "RandomForestRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "SupportVectorMachineRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("SVR rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')
    


    #LinearRegression rmse:  [16.7 12.9 11.6 14.  18.6]
    #Avg. rmse:  14.76
    
    #DecisionTreeRegressor rmse:  [13.1  8.4 10.8 14.  14.1]
    #Avg. rmse:  12.08
    
    #RandomForestRegressor rmse:  [ 9.6  7.2  8.1 12.7 13.8]
    #Avg. rmse:  10.28
    
    #SVR rmse:  [ 8.   8.1  7.  10.  14.4]
    #Avg. rmse:  9.5
    


    # => chọn RandomForestRegressor và SVR để tới bước fine tune model.

# In[6]: FINE-TUNE MODELS 
print('\n____________________________________ Fine-tune models ____________________________________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    print('Best estimator: ', grid_search.best_estimator_)  
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 




if run_new_search:
    # GridSearchCV cho RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    param_grid = [
        {'bootstrap': [True], 'n_estimators': [3, 30, 60], 'max_features': [2, 10, 25, 37]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 10, 25]} ]
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(processed_train_set_val, train_set_labels)
    joblib.dump(grid_search,'saved_objects/RandomForestRegressor_gridsearch.pkl')
    print_search_result(grid_search, model_name = "RandomForestRegressor") 
    # Best hyperparameter combination:  {'bootstrap': True, 'max_features': 10, 'n_estimators': 30}
    # Best rmse:  10.09488976664488
    # Best estimator:  RandomForestRegressor(max_features=10, n_estimators=30, random_state=42)

    model = SVR()
    param_grid = [
        {'kernel':['rbf'],'epsilon': [1,2,4], 'C': [10,100,10000]},]
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(processed_train_set_val, train_set_labels)
    joblib.dump(grid_search,'saved_objects/SVR_gridsearch.pkl')
    print_search_result(grid_search, model_name = "SVR") 
    # Best hyperparameter combination:  {'C': 10000, 'epsilon': 2, 'kernel': 'rbf'}
    # Best rmse:  10.815187945882526
    # Best estimator:  SVR(C=10000, epsilon=2)

#6.1 Ensemble models affter fine-tune
if new_run_ensemble:
    model_name = "VotingEnsembleRegressor" 
    r1 = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl').best_estimator_
    r2 = joblib.load('saved_objects/SVR_gridsearch.pkl').best_estimator_
    model = VotingRegressor([('rf', r1),('svr',r2)])
    model.fit(processed_train_set_val, train_set_labels)
    joblib.dump(model,'saved_objects/EnsembleFineTune_model.pkl')
    r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
    print('R2 score (on training data, best=1):', r2score)
    print("Root Mean Square Error: ", rmse.round(decimals=1))
    #Evaluate this Model
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("VotingEnsembleRegressor rmse: ", rmse_scores.round(decimals=1))
    # VotingEnsembleRegressor rmse:  [ 8.   7.7  7.6 10.8 13.5]

model_name = "VotingEnsembleRegressor" 
rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
print("SVR rmse: ", rmse_scores.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n') # Avg. rmse:  9.52

# => Solution là model Ensemble của các predictor sau quá trình Fine-tune.

a=1
# In[7]: ANALYZE AND TEST YOUR SOLUTION
# Load Ensemble Model Fine-tuned
best_model = joblib.load('saved_objects/EnsembleFineTune_model.pkl') 

# So sánh với random forest
#best_model = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl').best_estimator_

# So sánh với SVR
#best_model = joblib.load('saved_objects/SVR_gridsearch.pkl').best_estimator_


print('\n____________________________________ ANALYZE AND TEST YOUR SOLUTION ____________________________________')
print('SOLUTION: ' , best_model)
store_model(best_model, model_name="SOLUTION") 


# 7.3 Run on test data
processed_test_set = full_pipeline.transform(test_set)  

# 7.3.1 Compute R2 score and root mean squared error
r2score, rmse = r2score_and_rmse(best_model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

# 7.3.2 Predict labels for some test instances
print("\nTest data: \n", test_set.iloc[0:9])
print("Predictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')

#Test data:
#           HÃNG       DÒNG  NĂM_SX DUNG TÍCH - CC       KIỂU XE
#      HÃNG       DÒNG  NĂM_SX  DUNG TÍCH - CC       KIỂU XE TÌNH TRẠNG
#     Honda  Air Blade  2010.0           110.0        Tay ga         Cũ
#     Honda       Wave  2010.0           110.0         Xe số         Cũ
#   Piaggio    Liberty  2013.0           125.0        Tay ga         Cũ
#    Yamaha    Exciter  2012.0           125.0  Tay côn/Moto         Cũ
#     Honda       Wave  2008.0           110.0         Xe số         Cũ
#     Honda       Lead  2015.0           110.0        Tay ga         Cũ
#     Honda  Forza 300  2011.0           300.0        Tay ga         Cũ
#     Honda      Vario  2020.0           125.0        Tay ga         Cũ
#     Honda      Vario  2019.0           125.0        Tay ga         Cũ
    
########################            Performance on test data           ########################
# Với Ensemble VotingRegressor
# R2 score (on test data, best=1): 0.9830943635005641
# Root Mean Square Error:  7.9
# Predictions:  [17.4   9.2  17.5  19.9   9.2  24.1  154.   45.4  44.5]
# Labels:       [16.6,  7.0, 16.5, 25.9,  9.5, 24.8, 155.0, 42.0, 45.0]

# Với Random Forest
# Performance on test data:
# R2 score (on test data, best=1): 0.971480811409869
# Root Mean Square Error:  10.2
# Predictions:  [18.3   9.1  17.4  19.2   8.9  24.1  155.   45.   44.2]
# Labels:       [16.6, 7.0,  16.5, 25.9,  9.5, 24.8, 155.0, 42.0, 45.0]

# Với SVR(C=10000, epsilon=2)
# Performance on test data:
# R2 score (on test data, best=1): 0.98416494626391
# Root Mean Square Error:  7.6
# Predictions:  [16.5   9.2  17.5  20.5   9.5  24.   153.   45.8  44.8]
# Labels:       [16.6,  7.0, 16.5, 25.9,  9.5, 24.8, 155.0, 42.0, 45.0]



# Các bước trên đã được lặp đi lặp lại nhiều lần, trải qua quá trình chỉnh sửa loại bỏ nhiễu
# model tốt nhất hiện tại là SVR(C=10000, epsilon=2) voting ensemble kết hợp VCR và Random Forest cũng rất tốt MSE đạt 7.9

# DONE!
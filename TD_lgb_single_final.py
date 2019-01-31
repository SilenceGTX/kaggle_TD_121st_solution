# -*- coding: utf-8 -*-
'''
You can get a 0.9807 on public LB using this script(0.9812 on private LB).
Then, if you uses tr_8+tr_9 for training without any validation using the same parameters, 
and setting the num_round to 1.1× the best iteration round, you will get 0.9813 on public LB
(0.9821 on private LB).
'''
import pandas as pd
import numpy as np
import os
import gc
import lightgbm as lgb
import pickle

os.chdir('F:/datamining/TalkingData')


'''*******Read train data*******'''
tr_org=pd.read_csv('train.csv',parse_dates=['click_time']) #read original train dataset
del tr_org['attributed_time']

tr_org['day']=tr_org.click_time.dt.day.astype(np.uint32) #return day
tr_org['hour']=tr_org.click_time.dt.hour.astype(np.uint32) #return hour
tr_org['min']=tr_org.click_time.dt.minute.astype(np.uint32) #return minute
tr_org['sec']=tr_org.click_time.dt.second.astype(np.uint32) #return second
del tr_org['click_time']

tr_org['time']=tr_org['sec']+tr_org['min']*100+tr_org['hour']*10000+tr_org['day']*1000000
tr_org=tr_org[tr_org['time']>=6160000]
tr_org=tr_org[tr_org['time']<=9155959] #use Day6 16:00:00~Day 9 15:59:59 to form 3 complete days

pickle.dump(tr_org,open('./train_3days','wb'), protocol=4)


'''*******Time transformation*******'''
tr_7=tr_org[tr_org['time']<=7155959] #Day 6 16:00:00~Day 7 15:59:59, as 1st day

tr_8=tr_org[tr_org['time']<=8155959] #Day 7 16:00:00~Day 8 15:59:59, as 2nd day
tr_8=tr_8[tr_8['time']>=7160000]

tr_9=tr_org[tr_org['time']>=8160000] #Day 8 16:00:00~Day 9 15:59:59, as 3rd day

del tr_org, tr_7['day'], tr_8['day'], tr_9['day']
gc.collect()

tr_7['hour']=tr_7['hour'].apply(lambda x: (x+8)%24) #hour+8, transform into 0:00:00~23:59:59
tr_7['time']=tr_7['sec']+tr_7['min']*100+tr_7['hour']*10000
tr_8['hour']=tr_8['hour'].apply(lambda x: (x+8)%24) #hour+8, transform into 0:00:00~23:59:59
tr_8['time']=tr_8['sec']+tr_8['min']*100+tr_8['hour']*10000
tr_9['hour']=tr_9['hour'].apply(lambda x: (x+8)%24) #hour+8, transform into 0:00:00~23:59:59
tr_9['time']=tr_9['sec']+tr_9['min']*100+tr_9['hour']*10000
del tr_7['min'], tr_8['min'], tr_9['min'], tr_7['sec'], tr_8['sec'], tr_9['sec']

pickle.dump(tr_7,open('./tr_7_new','wb'), protocol=4)
pickle.dump(tr_8,open('./tr_8_new','wb'), protocol=4)
pickle.dump(tr_9,open('./tr_9_new','wb'), protocol=4)


'''*******TTI(attributed_time-click_time)*******'''
# "atime" means attributed_time, "ctime" means click_time, interval=atime-ctime
actime=pickle.load(open('./a_c_interval_seconds','rb')) 
ac_7=actime[actime['ctime']<=7155959] #Day 6 16:00:00~Day 7 15:59:59, as 1st day

ac_8=actime[actime['ctime']<=8155959] #Day 7 16:00:00~Day 8 15:59:59, as 2nd day
ac_8=ac_8[ac_8['ctime']>=7160000]

ac_9=actime[actime['ctime']>=8160000] #Day 8 16:00:00~Day 9 15:59:59, as 3rd day

del actime
gc.collect()

'''*******各种函数*******'''
def Feature_agg(df, features, target, agg_type):
    as_name=('_').join(features)+'_'+agg_type+'_'+target
    grp=df.groupby(features,as_index=False)[target].agg(agg_type).reset_index(drop=True).rename({target:as_name},axis=1)
    df=df.merge(grp,how='left',on=features)
    print(as_name,' added!')
    del as_name, grp
    gc.collect()
    return df

def Feature_nextclick(df, features, diff_type=-1):
    #the interval(second) between this click to the next click. If diff_type=1, it means to the previous click
    as_name=('_').join(features)+'_'+'nextclick'+str(diff_type)
    arr_slice = df[features].values
    df['nc_key']=np.ravel_multi_index(arr_slice.T,arr_slice.max(0)+1) #equals to .ngroup() in pandas
    df[as_name]=diff_type*df.groupby(['nc_key'], group_keys=False)['time'].diff(diff_type)
    print(as_name,' added!')  
    del as_name, df['nc_key'], arr_slice
    gc.collect()    
    return df

def Feature_cumcount(df, features):
    #cumcount(the 1st click, the 2nd click, the 3rd click...)
    as_name=('_').join(features)+'_'+'cumcount'
    df[as_name]=df.groupby(features,as_index=False).cumcount()
    print(as_name,' added!')
    del as_name
    gc.collect() 
    return df

def Feature_desc(df, features):
    #countdown(the 1st countdown, the 2nd countdown, the 3rd countdown...)
    as_name=('_').join(features)+'_'+'desc'
    df[as_name]=df.groupby(features,as_index=False).cumcount(ascending=False)
    print(as_name,' added!')
    del as_name
    gc.collect() 
    return df

def Feature_nunique(df, features, target):
    as_name=('_').join(features)+'_'+'nuni'+'_'+target
    gp=df.groupby(by=features)[target].nunique().reset_index().rename({target:as_name},axis=1)
    df=df.merge(gp,how='left',on=features)
    print(as_name,' added!')
    del as_name, gp
    gc.collect() 
    return df

def Feature_livetime(df, features):
    #last click time-first click time
    as_name=('_').join(features)+'_'+'livetime'
    arr_slice = df[features].values
    df['lt_key']=np.ravel_multi_index(arr_slice.T,arr_slice.max(0)+1)
    grp=df.groupby(['lt_key'], as_index=False)['time'].max().reset_index(drop=True).rename({'time':'lt_max'},axis=1)
    df=df.merge(grp, how='left', on='lt_key')
    grp=df.groupby(['lt_key'], as_index=False)['time'].min().reset_index(drop=True).rename({'time':'lt_min'},axis=1)
    df=df.merge(grp, how='left', on='lt_key')
    df[as_name]=df['lt_max']-df['lt_min']
    print(as_name,' added!')  
    del as_name, df['lt_key'], df['lt_max'], df['lt_min'], arr_slice
    gc.collect()    
    return df

def Feature_CTR(df1, df2, features):
    #calculate the conversion rate on df1 and merge it on df2
    as_name=('_').join(features)+'_'+'ctr'
    log_group = np.log(100000)
    grp=df1.groupby(features, as_index=False)['is_attributed'].mean().reset_index(drop=True).rename({'is_attributed':as_name},axis=1)
    cnt=df1.groupby(features, as_index=False)['is_attributed'].count().reset_index(drop=True).rename({'is_attributed':'cnt'},axis=1)
    cnt['cnt']=cnt['cnt'].apply(lambda x: np.min([1, np.log(x)/log_group]))
    grp['cnt']=cnt['cnt']   
    del cnt
    grp[as_name]=grp[as_name]*grp['cnt']
    del grp['cnt']
    df2=df2.merge(grp, how='left', on=features)
    print(as_name,' added!')  
    del as_name, grp
    gc.collect()    
    return df2   

def Group_feature_id(df, features, name):
    #use name to mark a goup of features
    arr_slice = df[features].values
    df[name]=np.ravel_multi_index(arr_slice.T,arr_slice.max(0)+1)
    del arr_slice
    gc.collect()
    return df

def DT_trans(feature, dtypes):
    #transform a feature's data type into dtypes
    global tr_8, tr_9
    tr_8[feature]=tr_8[feature].astype(dtypes)
    tr_9[feature]=tr_9[feature].astype(dtypes)
    gc.collect()
    return tr_8, tr_9


'''*******Feature engineering*******'''
# we use ip-device-os(ido) to mark a user
tr_8=Feature_nextclick(tr_8,['ip','device','os','app']) #the interval the user next click the app
tr_9=Feature_nextclick(tr_9,['ip','device','os','app'])

tr_8=Feature_nextclick(tr_8,['ip','device','os','app'], diff_type=1) #the interval the user previous click the app
tr_9=Feature_nextclick(tr_9,['ip','device','os','app'], diff_type=1) 

tr_8=Feature_cumcount(tr_8, ['ip','device','os','app']) #which times the user clicks the app
tr_9=Feature_cumcount(tr_9, ['ip','device','os','app'])

tr_8=Feature_nunique(tr_8, ['ip','device','os'],'app') #how many unique apps has the user clicked
tr_9=Feature_nunique(tr_9, ['ip','device','os'],'app')

tr_8=Feature_nunique(tr_8, ['ip'],'app') #how many unique apps has the ip clicked
tr_9=Feature_nunique(tr_9, ['ip'],'app') 

tr_8=Feature_livetime(tr_8, ['ip','device','os']) #user's lifetime
tr_9=Feature_livetime(tr_9, ['ip','device','os'])

tr_8=Feature_agg(tr_8, ['ip','device','os'], 'time', 'count') #how many times the user clicks
tr_9=Feature_agg(tr_9, ['ip','device','os'], 'time', 'count')

tr_8=Feature_agg(tr_8, ['ip'], 'time', 'count') #how many times the ip clicks
tr_9=Feature_agg(tr_9, ['ip'], 'time', 'count') 

tr_8=Feature_agg(tr_8, ['ip', 'hour'], 'time', 'count') #how many times the ip clicks in an hour
tr_9=Feature_agg(tr_9, ['ip', 'hour'], 'time', 'count') 


tr_8=Feature_agg(tr_8, ['ip','device','os'], 'ip_device_os_app_nextclick1', 'mean') #the mean interval of the user's click
tr_9=Feature_agg(tr_9, ['ip','device','os'], 'ip_device_os_app_nextclick1', 'mean') 

tr_8=Feature_agg(tr_8, ['ip', 'app'], 'time', 'count') #how many times the ip clicks the app
tr_9=Feature_agg(tr_9, ['ip', 'app'], 'time', 'count')

tr_8=Feature_CTR(tr_7, tr_8, ['app','channel']) #conversion rate of the app through the channel
tr_9=Feature_CTR(tr_8, tr_9, ['app','channel'])

tr_8=Feature_CTR(tr_7, tr_8, ['ip','device','os']) #conversion rate of the user clicks
tr_9=Feature_CTR(tr_8, tr_9, ['ip','device','os']) 

tr_8=Feature_CTR(tr_7, tr_8, ['app']) #conversion rate of the app
tr_9=Feature_CTR(tr_8, tr_9, ['app'])

tr_8=Feature_CTR(tr_7, tr_8, ['ip']) #conversion rate of the ip clicks
tr_9=Feature_CTR(tr_8, tr_9, ['ip']) 

tr_8['h15min']=tr_8['time']//100 
tr_8['h15min']=tr_8['h15min']%100
tr_8['h15min']=tr_8['h15min']//15
tr_8['hour']=tr_8['hour'].astype(np.uint32)
tr_8['h15min']=tr_8['hour']*10+tr_8['h15min']
tr_8=Feature_agg(tr_8, ['ip','device','os','app','h15min'], 'time', 'count') #how many times the user clicks the app in 15min
tr_8=Feature_nunique(tr_8, ['ip','device','os','h15min'], 'app') #how many unique apps has the user clicked in 15min
tr_8=Feature_agg(tr_8, ['ip','app','h15min'], 'time', 'count') #how many times the ip clicks the app in 15min
tr_8=Feature_nunique(tr_8, ['ip','h15min'], 'app') #how many unique apps has the ip clicked in 15min
del tr_8['h15min']
tr_9['h15min']=tr_9['time']//100
tr_9['h15min']=tr_9['h15min']%100
tr_9['h15min']=tr_9['h15min']//15
tr_9['hour']=tr_9['hour'].astype(np.uint32)
tr_9['h15min']=tr_9['hour']*10+tr_9['h15min']
tr_9=Feature_agg(tr_9, ['ip','device','os','app','h15min'], 'time', 'count')
tr_9=Feature_nunique(tr_9, ['ip','device','os','h15min'], 'app')
tr_9=Feature_agg(tr_9, ['ip','app','h15min'], 'time', 'count')
tr_9=Feature_nunique(tr_9, ['ip','h15min'], 'app')
del tr_9['h15min']

tr_8['h30min']=tr_8['time']//100 
tr_8['h30min']=tr_8['h30min']%100
tr_8['h30min']=tr_8['h30min']//30
tr_8['hour']=tr_8['hour'].astype(np.uint32)
tr_8['h30min']=tr_8['hour']*10+tr_8['h30min']
tr_8=Feature_agg(tr_8, ['ip','device','os','app','h30min'], 'time', 'count') #how many times the user clicks the app in 30min
tr_8=Feature_nunique(tr_8, ['ip','device','os','h30min'], 'app') #how many unique apps has the user clicked in 30min
del tr_8['h30min']
tr_9['h30min']=tr_9['time']//100
tr_9['h30min']=tr_9['h30min']%100
tr_9['h30min']=tr_9['h30min']//30
tr_9['hour']=tr_9['hour'].astype(np.uint32)
tr_9['h30min']=tr_9['hour']*10+tr_9['h30min']
tr_9=Feature_agg(tr_9, ['ip','device','os','app','h30min'], 'time', 'count')
tr_9=Feature_nunique(tr_9, ['ip','device','os','h30min'], 'app')
del tr_9['h30min']

tr_8=Feature_nunique(tr_8, ['ip'],'hour') #how many unique hours does the ip appear in
tr_9=Feature_nunique(tr_9, ['ip'],'hour')

tr_8=Feature_nunique(tr_8, ['app'],'channel') #the clicks of the app come from how many unique channels
tr_9=Feature_nunique(tr_9, ['app'],'channel') 

TTI_median7=ac_7.groupby(['app','ip'],as_index=False)['interval'].median() #the median click-attributed interval of the ip install the app
tr_8=tr_8.merge(TTI_median7, how='left', on=['app', 'ip']) #merge it on tr_8
del TTI_median7
gc.collect()
TTI_median8=ac_8.groupby(['app','ip'],as_index=False)['interval'].median() 
tr_9=tr_9.merge(TTI_median8, how='left', on=['app', 'ip']) #merge it on tr_9
del TTI_median8
gc.collect() 

del tr_7
#RAM limited，train on 8, val on 9
#tr_7=pickle.load(open('./tr_7_new_fea','rb')) 
#tr_8=pickle.load(open('./tr_8_new_fea','rb')) 
#tr_9=pickle.load(open('./tr_9_new_fea','rb')) 

tr_8, tr_9=DT_trans('ip', np.uint32)
tr_8, tr_9=DT_trans('app', np.uint16)
tr_8, tr_9=DT_trans('device', np.uint16)
tr_8, tr_9=DT_trans('os', np.uint16)
tr_8, tr_9=DT_trans('channel', np.uint16)
tr_8, tr_9=DT_trans('is_attributed', np.uint8)
tr_8, tr_9=DT_trans('hour', np.uint8)
tr_8, tr_9=DT_trans('time', np.uint32)

tr_8, tr_9=DT_trans('ip_device_os_app_desc', np.uint16)
tr_8, tr_9=DT_trans('ip_device_os_nuni_app', np.uint16)
tr_8, tr_9=DT_trans('ip_device_os_livetime', np.uint32)
tr_8, tr_9=DT_trans('ip_device_os_count_time', np.uint32)
tr_8, tr_9=DT_trans('ip_hour_count_time', np.uint16)
tr_8, tr_9=DT_trans('ip_count_time', np.uint32)
tr_8, tr_9=DT_trans('ip_nuni_app', np.uint16)
tr_8, tr_9=DT_trans('ip_app_count_time', np.uint32)
tr_8, tr_9=DT_trans('ip_device_os_app_h15min_count_time', np.uint16)
tr_8, tr_9=DT_trans('ip_device_os_h15min_nuni_app', np.uint8)
tr_8, tr_9=DT_trans('ip_app_h15min_count_time', np.uint16)
tr_8, tr_9=DT_trans('ip_h15min_nuni_app', np.uint8)
tr_8, tr_9=DT_trans('ip_device_os_app_h30min_count_time', np.uint16)
tr_8, tr_9=DT_trans('ip_device_os_h30min_nuni_app', np.uint8)
tr_8, tr_9=DT_trans('ip_nuni_hour', np.uint8)
tr_8, tr_9=DT_trans('app_nuni_channel', np.uint8)

pickle.dump(tr_8,open('./tr_8_new_fea','wb'), protocol=4)
pickle.dump(tr_9,open('./tr_9_new_fea','wb'), protocol=4)

#tr_7=pickle.load(open('./tr_7_new_fea','rb')) 
#tr_8=pickle.load(open('./tr_8_new_fea','rb')) 
#tr_9=pickle.load(open('./tr_9_new_fea','rb')) 


'''*******Validation(tr_9)*******'''
#only use time appeared in the test
r1=tr_9[tr_9.hour.isin([12,13])]
r2=tr_9[tr_9.time==140000]
r3=tr_9[tr_9.hour.isin([17,18])]
r4=tr_9[tr_9.time==190000]
r5=tr_9[tr_9.hour.isin([21,22])]
r6=tr_9[tr_9.time==230000]
del tr_9
val=pd.concat([r1, r2, r3, r4, r5, r6])
del r1, r2, r3, r4, r5, r6
gc.collect()


'''*******LGB training*******'''
lgb_tr=lgb.Dataset(tr_8.drop(['ip','is_attributed'],axis=1), label=tr_8['is_attributed'].values) 
lgb_val=lgb.Dataset(val.drop(['ip','is_attributed'],axis=1), label=val['is_attributed'].values)

params = {
            'learning_rate': 0.02,
            'num_leaves': 63,  # we should let it be smaller than 2^(max_depth)
            'max_depth': 6, 
            'min_child_samples': 100,  
            'max_bin': 100, 
            'subsample': 0.7, 
            'subsample_freq': 1,  
            'colsample_bytree': 0.7, 
            'min_child_weight': 0, 
            'scale_pos_weight':200, 
            'objective':'binary', 
            'lambda_l2':100,
            'metric':['auc']
} 

num_round =802
categorical = ['app', 'device', 'os', 'channel', 'hour']
evals_results={}

bst=lgb.train(params, lgb_tr, num_round, evals_result=evals_results, categorical_feature=categorical, valid_sets=[lgb_val]) #categorical_feature=categorical
best_num=evals_results['valid_0']['auc'].index(max(evals_results['valid_0']['auc']))+1
print('best round is:',best_num,'best AUC is:',max(evals_results['valid_0']['auc']))

best_num_round =best_num
bst=lgb.train(params, lgb_tr, best_num_round, categorical_feature=categorical, valid_sets=[lgb_val])
bst.save_model('re5_0507.model')
print(dict(zip(bst.feature_name(),bst.feature_importance())))

lgb.plot_importance(bst, max_num_features=10)
lgb.plot_importance(bst, importance_type='gain', max_num_features=10) #plot the feature importance, top10 of split & gain
del tr_8, val
gc.collect()


'''*******Read  test supplement*******'''
te_org=pd.read_csv('test_old.csv',parse_dates=['click_time']) #read test supplement dataset

te_org['day']=te_org.click_time.dt.day.astype(np.uint32) #return day
te_org['hour']=te_org.click_time.dt.hour.astype(np.uint32) #return hour
te_org['min']=te_org.click_time.dt.minute.astype(np.uint32) #return minute
te_org['sec']=te_org.click_time.dt.second.astype(np.uint32) #return second
del te_org['click_time']

te_org['time']=te_org['sec']+te_org['min']*100+te_org['hour']*10000+te_org['day']*1000000
te_org=te_org[te_org['time']>=9160000]
te_org=te_org[te_org['time']<=10155959] #Day9 16:00:00~Day10 15:59:59
del te_org['min'], te_org['sec'], te_org['click_id']

te_org=Feature_cumcount(te_org, ['ip','app','device','os','channel','time']) #cumcount is used to mark duplicate records in 1s 
te_org['merge_key']=te_org['ip'].astype(str)+'_'+te_org['app'].astype(str)+'_'+te_org['device'].astype(str)+\
'_'+te_org['os'].astype(str)+'_'+te_org['channel'].astype(str)+'_'+te_org['time'].astype(str)+'_'+\
te_org['ip_app_device_os_channel_time_cumcount'].astype(str) #merge_key is used to give every record an unique mark

te_online=pd.read_csv('./test.csv',parse_dates=['click_time']) #read online test dataset
te_online['day']=te_online.click_time.dt.day.astype(np.uint32) #return day
te_online['hour']=te_online.click_time.dt.hour.astype(np.uint32) #return hour
te_online['min']=te_online.click_time.dt.minute.astype(np.uint32) #return minute
te_online['sec']=te_online.click_time.dt.second.astype(np.uint32) #return second
del te_online['click_time']
te_online['time']=te_online['sec']+te_online['min']*100+te_online['hour']*10000+te_online['day']*1000000
del te_online['min'], te_online['sec']

te_online=Feature_cumcount(te_online, ['ip','app','device','os','channel','time']) 
te_online['merge_key']=te_online['ip'].astype(str)+'_'+te_online['app'].astype(str)+'_'+te_online['device'].astype(str)+\
'_'+te_online['os'].astype(str)+'_'+te_online['channel'].astype(str)+'_'+te_online['time'].astype(str)+'_'+\
te_online['ip_app_device_os_channel_time_cumcount'].astype(str) #merge_key is used to give every record an unique mark
te_online=te_online[['click_id', 'merge_key']]

te_org=te_org.merge(te_online, how='left', on='merge_key') #we can align the data in test supplement with test using merge_key
te_org=te_org.reset_index(drop=True)
del te_org['merge_key']
te_org['click_id']=te_org['click_id'].astype(int)
del te_online
gc.collect()

pickle.dump(te_org,open('./test_sup_org','wb'), protocol=4)

te_org['hour']=te_org['hour'].apply(lambda x: (x+8)%24) #hour+8, transform into 0:00:00~23:59:59
te_org['time']=te_org['time'].apply(lambda x: x%10000)
te_org['time']=te_org['time']+te_org['hour']*10000
del te_org['ip_app_device_os_channel_time_cumcount'], te_org['day']
pickle.dump(te_org,open('./test_sup_timed','wb'), protocol=4)



'''*******Test feature engineering*******'''
#te_org=pickle.load(open('./test_sup_timed','rb')) 
tr_9=pickle.load(open('./tr_9_new_fea','rb')) 

#te_org=pickle.load(open('./test_sup_featured','rb')) 
te_org=Feature_nextclick(te_org,['ip','device','os','app'])
te_org=Feature_nextclick(te_org,['ip','device','os','app'], diff_type=1)
te_org=Feature_cumcount(te_org, ['ip','device','os','app'])
te_org=Feature_nunique(te_org, ['ip','device','os'],'app')
te_org=Feature_nunique(te_org, ['ip'],'app')
te_org=Feature_livetime(te_org, ['ip','device','os'])
te_org=Feature_agg(te_org, ['ip','device','os'], 'time', 'count')
te_org=Feature_agg(te_org, ['ip'], 'time', 'count')
te_org=Feature_agg(te_org, ['ip', 'hour'], 'time', 'count')
te_org=Feature_agg(te_org, ['ip','device','os'], 'ip_device_os_app_nextclick1', 'mean')
te_org=Feature_agg(te_org, ['ip', 'app'], 'time', 'count')
te_org=Feature_CTR(tr_9, te_org, ['app','channel'])
te_org=Feature_CTR(tr_9, te_org, ['ip','device','os'])
te_org=Feature_CTR(tr_9, te_org, ['app'])
te_org=Feature_CTR(tr_9, te_org, ['ip'])

te_org['h15min']=te_org['time']//100 
te_org['h15min']=te_org['h15min']%100
te_org['h15min']=te_org['h15min']//15
te_org['hour']=te_org['hour'].astype(np.uint32)
te_org['h15min']=te_org['hour']*10+te_org['h15min']
te_org=Feature_agg(te_org, ['ip','device','os','app','h15min'], 'time', 'count') 
te_org=Feature_nunique(te_org, ['ip','device','os','h15min'], 'app')
te_org=Feature_agg(te_org, ['ip','app','h15min'], 'time', 'count')
te_org=Feature_nunique(te_org, ['ip','h15min'], 'app') 
del te_org['h15min'] 

te_org['h30min']=te_org['time']//100 
te_org['h30min']=te_org['h30min']%100
te_org['h30min']=te_org['h30min']//30
te_org['hour']=te_org['hour'].astype(np.uint32)
te_org['h30min']=te_org['hour']*10+te_org['h30min']
te_org=Feature_agg(te_org, ['ip','device','os','app','h30min'], 'time', 'count') #30min level，num of clicks on apps
te_org=Feature_nunique(te_org, ['ip','device','os','h30min'], 'app') #30min level，num of clicks on unique apps
del te_org['h30min'] 

te_org=Feature_nunique(te_org, ['ip'],'hour')
te_org=Feature_nunique(te_org, ['app'],'channel') 

TTI_median9=ac_9.groupby(['app','ip'],as_index=False)['interval'].median() 
te_org=te_org.merge(TTI_median9, how='left', on=['app', 'ip'])
del TTI_median9
gc.collect()


te_org['ip']=te_org['ip'].astype(np.uint32)
te_org['app']=te_org['app'].astype(np.uint16)
te_org['device']=te_org['device'].astype(np.uint16)
te_org['os']=te_org['os'].astype(np.uint16)
te_org['channel']=te_org['channel'].astype(np.uint16)
te_org['is_attributed']=te_org['is_attributed'].astype(np.uint8)
te_org['hour']=te_org['hour'].astype(np.uint8)
te_org['time']=te_org['time'].astype(np.uint32)

te_org['ip_device_os_app_desc']=te_org['ip_device_os_app_desc'].astype(np.uint16)
te_org['ip_device_os_nuni_app']=te_org['ip_device_os_nuni_app'].astype(np.uint16)
te_org['ip_device_os_livetime']=te_org['ip_device_os_livetime'].astype(np.uint32)
te_org['ip_device_os_count_time']=te_org['ip_device_os_count_time'].astype(np.uint32)
te_org['ip_hour_count_time']=te_org['ip_hour_count_time'].astype(np.uint16)
te_org['ip_count_time']=te_org['ip_count_time'].astype(np.uint32)
te_org['ip_nuni_app']=te_org['ip_nuni_app'].astype(np.uint16) 
te_org['ip_app_count_time']=te_org['ip_app_count_time'].astype(np.uint32)
te_org['ip_device_os_app_h15min_count_time']=te_org['ip_device_os_app_h15min_count_time'].astype(np.uint16)
te_org['ip_device_os_h15min_nuni_app']=te_org['ip_device_os_h15min_nuni_app'].astype(np.uint8) 
te_org['ip_app_h15min_count_time']=te_org['ip_app_h15min_count_time'].astype(np.uint16) 
te_org['ip_h15min_nuni_app']=te_org['ip_h15min_nuni_app'].astype(np.uint8) 
te_org['ip_device_os_app_h30min_count_time']=te_org['ip_device_os_app_h30min_count_time'].astype(np.uint16) 
te_org['ip_device_os_h30min_nuni_app']=te_org['ip_device_os_h30min_nuni_app'].astype(np.uint8) 
te_org['ip_nuni_hour']=te_org['ip_nuni_hour'].astype(np.uint8) 
te_org['app_nuni_channel']=te_org['app_nuni_channel'].astype(np.uint8) 

del tr_9
pickle.dump(te_org,open('./test_sup_featured','wb'), protocol=4)
gc.collect()


'''*******Prediction*******'''
te_org=te_org[te_org.click_id.notnull()==True]
lgb_te=lgb.Dataset(te_org.drop(['ip','click_id'],axis=1))
pred=bst.predict(lgb_te.data)

res=te_org[['click_id']]
res['is_attributed']=pred
res['click_id']=res['click_id'].astype(int)

submit=pd.read_csv('test_id.csv')
submit=submit.merge(res, how='left', on='click_id')
print(submit[:10])
submit.to_csv('re5_lgb_0507.csv',index=False)

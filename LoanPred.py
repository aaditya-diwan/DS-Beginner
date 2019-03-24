#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the datasets
df=pd.read_csv('Train.csv',delimiter=',')
test_df=pd.read_csv('Test.csv',delimiter=',')
df=df.drop('Loan_ID',axis=1)
test_df=test_df.drop('Loan_ID',axis=1)

#Train missing values
df['LoanAmount'].fillna(value=146.0, inplace= True)
df['Loan_Amount_Term'].fillna(342.0,inplace= True)
df['Married'].fillna('Yes',inplace= True)
df['Credit_History'].fillna(1.0,inplace= True)
df['Self_Employed'].fillna('No',inplace= True)
df['Gender'].fillna('Male',inplace= True)
df['Dependents'].fillna('0',inplace= True)


#Test missing values
test_df['LoanAmount'].fillna(136, inplace= True)
test_df['Loan_Amount_Term'].fillna(342,inplace= True)
test_df['Credit_History'].fillna(1.0,inplace= True)
test_df['Self_Employed'].fillna('No',inplace= True)
test_df['Gender'].fillna('Male',inplace= True)
test_df['Dependents'].fillna('0',inplace= True)

#taking care of outliers
df['LoanAmount_log']= np.log(df['LoanAmount'])
test_df['LoanAmount_log']= np.log(test_df['LoanAmount'])

df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_Log']=np.log(df['TotalIncome'])


test_df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
test_df['TotalIncome_Log']=np.log(df['TotalIncome'])


test_df= test_df.drop(['ApplicantIncome','CoapplicantIncome','LoanAmount','TotalIncome'],axis=1)
df= df.drop(['ApplicantIncome','CoapplicantIncome','LoanAmount','TotalIncome'],axis=1)

# Making all of them as int64
#df.describe()
#df['CoapplicantIncome']= df['CoapplicantIncome'].astype(np.int64)
#df['LoanAmount']= df['LoanAmount'].astype(np.int64)
df['Loan_Amount_Term']= df['Loan_Amount_Term'].astype(np.int64)
df['Credit_History']= df['Credit_History'].astype(np.int64)
df['LoanAmount_log']= df['LoanAmount_log'].astype(np.int64)
#df['TotalIncome']= df['TotalIncome'].astype(np.int64)
df['TotalIncome_Log']= df['TotalIncome_Log'].astype(np.int64)

#test data
# Making all of them as int64

#test_df['CoapplicantIncome']= test_df['CoapplicantIncome'].astype(np.int64)
#test_df['LoanAmount']= test_df['LoanAmount'].astype(np.int64)
test_df['Loan_Amount_Term']= test_df['Loan_Amount_Term'].astype(np.int64)
test_df['Credit_History']= test_df['Credit_History'].astype(np.int64)
test_df['LoanAmount_log']= test_df['LoanAmount_log'].astype(np.int64)
#test_df['TotalIncome']= test_df['TotalIncome'].astype(np.int64)
test_df['TotalIncome_Log']= test_df['TotalIncome_Log'].astype(np.int64)


from sklearn.preprocessing import LabelEncoder

var= ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le= LabelEncoder() # Object
for i in var:
    df[i]= le.fit_transform(df[i])
    if (i!='Loan_Status'):
        test_df[i]= le.fit_transform(test_df[i])
        
        
df['result']= df['Loan_Status']
df= df.drop(['Loan_Status'],axis= 1)     


X= df.iloc[:,1:-1].values
y= df.iloc[:,-1].values

#Split the data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size= 0.3,random_state= 5)   



# fit model
from sklearn.ensemble import RandomForestClassifier 
classifier= RandomForestClassifier(n_estimators= 700)
classifier.fit(X_train,y_train)
pred= classifier.predict(X_test)

# For solution
X_t = test_df.iloc[:,1:].values
#pre= classifier.predict(X_t)
X_t.shape
pre= classifier.predict(X_t)
#test_df['Loan_Status']= predict.astype(np.int64)
#test_df['Loan_Status']=test_df['Loan_Status'].map({0:'N',1:'Y'})
#test_df.to_csv('submission.csv',index= false)

from sklearn import metrics as m
# Accuracy
acc= m.accuracy_score(y_test,pred)
acc
test_df['Loan_Status']= pre


test_df['Loan_Status']= test_df['Loan_Status'].map({0:'N',1:'Y'})

:
test_df= test_df.drop(['Gender','Married','Dependents','Education','Self_Employed','Loan_Amount_Term','Credit_History','Property_Area','LoanAmount_log','TotalIncome_log'],axis=1)

test_df.to_csv('sample_submission.csv',index= False)
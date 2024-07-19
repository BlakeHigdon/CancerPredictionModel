# Student Name: Blake Higdon
# Student Number: 0110015090
# Program Mentor: Maria Rozario

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import joblib

# load in the data that was found on Kaggle
rawCancerData = pd.read_csv('The_Cancer_data_1500_V2.csv')
# print(rawCancerData.head())

# Now I am going to create my X and Y values as the features (X) and prediction (Y)
x = rawCancerData[['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk',
                   'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']]

y = rawCancerData['Diagnosis']

# split data for testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y)

# use the libraries to standardize the features (x)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# time to actually train the model using are libraries
cancerModel = LogisticRegression()
cancerModel.fit(x_train, y_train)

# now collect stats and make predictions such as accuracy
# predict the patient having cancer or not
y_pred = cancerModel.predict(x_test)

# predict model accuracy
modelAccuracy = accuracy_score(y_test, y_pred)
# print(f'Model Accuracy:\n{modelAccuracy}')

# confusion matrix
modelConfusionMatrix = confusion_matrix(y_test, y_pred)
confusion_dataframe = pd.DataFrame(modelConfusionMatrix, index=['No Cancer', 'Cancer'], columns=['No Cancer', 'Cancer'])

fig = px.imshow(confusion_dataframe, text_auto=True, color_continuous_scale='Blues', title='Confusion Matrix')
fig.update_layout(xaxis_title='Predicted', yaxis_title='Actual')
fig.show()
print(f'Confusion Matrix:\n{modelConfusionMatrix}')
# (Plotly Technologies Inc., n.d.)

# model classification report
cReport = classification_report(y_test, y_pred, output_dict=True)
report_dataframe = pd.DataFrame(cReport).transpose()
if 'support' in report_dataframe.columns:
    report_dataframe = report_dataframe.drop(columns=['support'])

fig = go.Figure()


classes = report_dataframe.index[:-3]
for metric in ['precision', 'recall', 'f1-score']:
    fig.add_trace(go.Bar(
        x=classes,
        y=report_dataframe.loc[classes, metric],
        name=metric,
        text=report_dataframe.loc[classes, metric].round(2),
        textposition='auto'
    ))


fig.update_layout(
    title='Classification Report Metrics',
    xaxis_title='Class',
    yaxis_title='Score',
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    template='plotly_white'
)

fig.show()
# print(f'Classification Report:\n{cReport}')
# (Plotly Technologies Inc., n.d.)'''


# save and load the model for the program to reference
joblib.dump(cancerModel, 'cancer_model.pk1')

cModel = joblib.load('cancer_model.pk1')

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, cancerModel.predict_proba(x_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Create ROC curve plot
fig = go.Figure()

fig.add_trace(go.Scatter(x=fpr, y=tpr,
                         mode='lines',
                         name=f'ROC curve (area = {roc_auc:.2f})',
                         line=dict(color='darkorange', width=2)))

fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                         mode='lines',
                         line=dict(color='navy', width=2, dash='dash'),
                         showlegend=False))

fig.update_layout(title='Receiver Operating Characteristic',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate',
                  width=800, height=600,
                  template='plotly_white')

fig.show()
# (Plotly Technologies Inc., n.d.)


# create the interface in the console for the user to input their values

def usr_submitted_cancer_data():
    age = int(input('Enter your age (20-80) : '))
    while (age > 80) or (age < 20):
        print('Invalid input')
        age = int(input('Enter your age (20-80) : '))

    gender = int(input("Enter Gender (0 for Male, 1 for Female): "))
    while (gender != 0) and (gender != 1):
        print('Invalid input')
        gender = int(input("Enter Gender (0 for Male, 1 for Female): "))

    bmi = float(input("Enter BMI (15-40): "))
    while (bmi < 15) or (bmi > 40):
        print('Invalid input')
        bmi = float(input("Enter BMI (15-40): "))

    smoking = int(input("Do you smoke? (0 for No, 1 for Yes): "))
    while (smoking != 0) and (smoking != 1):
        print('Invalid input')
        smoking = int(input("Do you smoke? (0 for No, 1 for Yes): "))

    genetic_risk = int(input("Enter Your Genetic Risk to Cancer (0 for Low, 1 for Medium, 2 for High): "))
    while (genetic_risk != 0) and (genetic_risk != 1) and (genetic_risk != 2):
        print('Invalid input')
        genetic_risk = int(input("Enter Your Genetic Risk to Cancer (0 for Low, 1 for Medium, 2 for High): "))

    physical_activity = float(input("Enter Physical Activity (hours per week, 0-10): "))
    while (physical_activity < 0) or (physical_activity > 10):
        print('Invalid input')
        physical_activity = float(input("Enter Physical Activity (hours per week, 0-10): "))

    alcohol_intake = float(input("Enter Alcohol Intake in units per week ranging from 0-5 "
                                 "(1 unit is defined as 10ml/8g of pure alcohol): "))
    while (alcohol_intake > 5) or (alcohol_intake < 0):
        print('Invalid input')
        alcohol_intake = float(input("Enter Alcohol Intake (units per week, 0-5): "))

    cancer_history = int(input("Do you have a personal history of cancer? (0 for No, 1 for Yes): "))
    while (cancer_history != 0) and (cancer_history != 1):
        print('Invalid input')
        cancer_history = int(input("Do you have a personal history of cancer? (0 for No, 1 for Yes): "))

    usr_data = {
        'Age': [age],
        'Gender': [gender],
        'BMI': [bmi],
        'Smoking': [smoking],
        'GeneticRisk': [genetic_risk],
        'PhysicalActivity': [physical_activity],
        'AlcoholIntake': [alcohol_intake],
        'CancerHistory': [cancer_history]
    }

    return pd.DataFrame(usr_data)


# get input and scale it
usr_data = usr_submitted_cancer_data()

usr_submitted_data_scaled = scaler.transform(usr_data)

# have the model predict if patient has cancer or not

cancer_prediction = cModel.predict(usr_submitted_data_scaled)

# output result to user
if cancer_prediction[0] == 1:
    print("It is predicted that you have cancer. Please see your medical professional for further verification.")
else:
    print("It is predicted that you do not have cancer. "
          "If you have any worries/concerns, please see your medical professional.")





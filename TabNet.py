import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

clf = TabNetClassifier(
    optimizer_params=dict(lr=1e-3) 
)

df = pd.read_csv('/Users/chan050714/space/star_classification.csv')
drop_cols = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 
             'field_ID', 'spec_obj_ID', 'plate', 'MJD', 'fiber_ID']
df = df.drop(columns=drop_cols)

X = df.drop(columns=['class'])
y = df['class']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
classes = label_encoder.classes_

X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

clf = TabNetClassifier()

print("\n가장 기본적인 TabNet 학습을 시작합니다...")
clf.fit(
    X_train=X_train_scaled, y_train=y_train,
    eval_set=[(X_val_scaled, y_val)],
    eval_name=['valid'],
    eval_metric=['accuracy'],
    max_epochs=100,
    patience=10,          
    batch_size=1024,      
    virtual_batch_size=128
)

preds = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, preds)

print(f"\n최종 테스트 정확도 (Test Accuracy): {acc:.4f}")
print("\n상세 분류 리포트:")
print(classification_report(y_test, preds, target_names=classes))

feature_importances = clf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\n[ 피처 중요도 (Feature Importance) ]")
print(importance_df)
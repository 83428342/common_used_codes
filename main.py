## 라이브러리 및 메소드

# 데이터 조작용 라이브러리

import pandas as pd
import numpy as np

# 데이터 시각화용 라이브러리

import matplotlib.pyplot as plt
import matplotlib.image as mpimg # 이미지 파일 다룰때 사용
import seaborn as sns
%matplotlib inline # jupyter notobook용 명령어. 별도의 창이 아닌 셀 안에서 직접 그림이 표시되게 함.

sns.set(style='white', context='notebook', palette='deep') # seaborn 이미지 스타일 조정

# 데이터 분할용 라이브러리

from sklearn.model_selection import train_test_split

# one-hot encoding용 메소드

from tensorflow.keras.utils import to_categorical # 시각적 정답 레이블의 벡터화를 위해 사용

# 모델 전처리용 메소드

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 모델 구축용 라이브러리 및 메소드

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

# 모델 학습용 메소드

from tensorflow.keras.optimizers import Adam

# 모델 학습률 개선용 라이브러리

from tensorflow.keras.callbacks import ReduceLROnPlateau # 정확도가 높아지면 학습률을 매우 섬세하게 조정
from tensorflow.keras.callbacks import EarlyStopping # 검증 데이터 정확도 낮아지기 시작하면 중지

# 학습 결과 분류 라이브러리

from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------------------------------------

## 데이터 조작 및 확인

# Load the data, 경로는 다를 수 있음
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# 데이터 확인

train.head()
test.head()

# data 의 학습 요소와 정답 분리

Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)

# 더이상 쓰지 않는 대규모 데이터 메모리 해제

del train 

# 정답 레이블 별 개수 시각화하는 seaborn 함수

g = sns.countplot(x=Y_train)

# 정답 레이블 별 개수 출력하는 pandas 함수

Y_train.value_counts()

# 결측치 확인

X_train.isnull().any().describe()

# 이미지 csv 파일 학습가능한 형태로 배열 변환

X_train = X_train.values.reshape(-1, 28, 28, 1) # -1은 데이터 개수에 맞춰 자동으로 변환해줄때 사용
test = test.values.reshape(-1, 28, 28, 1) # 흑백 데이터는 RBG가 1개의 체널이기 때문에 마지막에 1 사용

# 정규화

# 이미지 데이터 정규화

X_train = X_train / 255.0
test = test / 255.0

# 정규분포를 따르는 데이터 정규화

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Batch Normalization

from tensorflow.keras.layers import BatchNormalization

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())

# one-hot-encoding으로 분류 가능하게 정답 레이블 벡터화

Y_train = to_categorical(Y_train, num_classes = 10)
Y_val = to_categorical(Y_val, num_classes = 10)

# train set과 validation set 분류

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

# 데이터 배열 변환 후의 이미지 데이터 확인

g = plt.imshow(X_train[0][:,:,0])

# ------------------------------------------------------------------------------------------------

## 신경망 모델 구축

# 기본 모델 정의

model = Sequential()

# input layer 구축

from tensorflow.keras.layers import Input

model.add(Input(shape=(28, 28, 1)))

# 완전 연결 층 구축

model.add(Dense(256, activation = "relu"))

# 다중 분류를 위한 출력층 구축

model.add(Dense(10, activation = "softmax")) # softmax 함수를 이용하여 다중 레이블 분류

## CNN 모델 구축

# Convolution layer 구축

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
     activation='relu', input_shape=(28,28,1)))   

# Pooling layer 구축

model.add(MaxPool2D(pool_size=(2,2)))

# Conv layer과 Pooling layer이후 완전 연결 층으로 가기 전 배열 평탄화

model.add(Flatten())

# Dropout으로 과적합 방지

model.add(Dropout(0.25))

# ------------------------------------------------------------------------------------------------

## model optimize

# 기본 optimizer로 Adam 사용

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# 정확도 높아지면 세세하게 학습, 그러나 학습률이 고원 상태이면 문제가 발생할 수 있음

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
    patience=3,                                              
    verbose=1,                                               
    factor=0.5, 
    min_lr=0.00001
    )

# EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# ------------------------------------------------------------------------------------------------

## Data augmentation

datagen = ImageDataGenerator(
    rotation_range=10,  # 무작위 회전 범위 (0~10도)
    zoom_range=0.1,  # 무작위 확대 범위
    width_shift_range=0.1,  # 무작위로 가로 방향으로 이미지 이동 (전체 너비의 10%)
    height_shift_range=0.1,  # 무작위로 세로 방향으로 이미지 이동 (전체 높이의 10%)
    horizontal_flip=False,  # 수평 방향으로 이미지를 무작위로 뒤집지 않음
    vertical_flip=False  # 수직 방향으로 이미지를 무작위로 뒤집지 않음
    )

datagen.fit(X_train)

# ------------------------------------------------------------------------------------------------

## 모델 학습

history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=batch_size),
    epochs=epochs, 
    validation_data=(X_val, Y_val),
    verbose=1, 
    steps_per_epoch=X_train.shape[0] // batch_size,
    callbacks=[learning_rate_reduction]
    )

# ------------------------------------------------------------------------------------------------

## 모델 평가

# loss와 accuracy 그래프 출력

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")
ax[0].set_title('Model Loss') 
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
ax[1].set_title('Model Accuracy') 
legend = ax[1].legend(loc='best', shadow=True)

plt.tight_layout()
plt.show()

# EarlyStopping을 사용한 경우의 정확도 출력

val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0) # EarlyStopping 후 최종 평가
print(f"Validation Accuracy after EarlyStopping: {val_acc:.4f}")

best_val_acc = max(history.history['val_accuracy']) # 가장 높은 검증 정확도 출력
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# EarlyStopping을 사용하지 않은 경우의 정확도 출력

val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0) # 학습 종료 후 최종 평가
print(f"Validation Accuracy after training: {val_acc:.4f}")

best_val_acc = max(history.history['val_accuracy']) # 가장 높은 검증 정확도 출력
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# confusion matrix

from sklearn.metrics import confusion_matrix

Y_pred = model.predict(X_val) # 검증 데이터에 대해 예측 수행

Y_pred_classes = np.argmax(Y_pred, axis=1) # 예측한 클래스 라벨로 변환 (원-핫 인코딩에서 클래스 인덱스로 변환)

Y_true = np.argmax(Y_val, axis=1) # 실제 클래스 라벨 (원-핫 인코딩에서 클래스 인덱스로 변환)

conf_matrix = confusion_matrix(Y_true, Y_pred_classes) # 혼동 행렬 계산

plt.figure(figsize=(8,6)) # 혼동 행렬 시각화
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

# CNN 모델의 틀린 출력 결과 시각화

errors = (Y_pred_classes - Y_true != 0) # 오류 선택
Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index, img_errors, pred_errors, obs_errors): #  예측과 실제 값이 다른 상위 6개 이미지를 보여주는 함수
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
            ax[row, col].set_title("Predicted label: {}\nTrue label: {}".format(pred_errors[error], obs_errors[error]))
            n += 1

Y_pred_errors_prob = np.max(Y_pred_errors, axis=1) # 오류 확률 계산
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

most_important_errors = sorted_dela_errors[-6:] # 오차가 큰 상위 6개 샘플 출력
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

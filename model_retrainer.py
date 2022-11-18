# imports
import pandas as pd
import matplotlib.pyplot as plt #Отрисовка графиков
from tensorflow.keras.utils import to_categorical
import numpy as np #Numpy
from tensorflow.keras.optimizers import Adam #Оптимизатор
from tensorflow.keras.models import Sequential, load_model #model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv1D, Conv2D, LSTM, MaxPooling1D #слои
from tensorflow.keras.metrics import Recall
from tensorflow import device
from tensorflow.keras.callbacks import ModelCheckpoint



def predict_crop(x):
    return np.argmax(model.predict(x))


train_df = pd.read_csv("./train_dataset_train.csv") #read train dataset
train_df = train_df.sort_index(axis=1)# preprocessing, sort by time
#create train dataset
train_df_nn = train_df.drop(["id", ".geo", "area"], axis = 1)
#data preprocessing

print(len(train_df_nn))

x = []

y = []

for f in range(len(train_df_nn)):
    x+=  [train_df_nn.iloc[f].to_numpy()[1:-1]]
    y += [train_df_nn.iloc[f].to_numpy()[0]]
x = np.array(x)
y = np.array(y)
#print(x[0],y, len(x), len(y), sep='\n')
#создаем обучающую и тестовую выборку для нейронки
x_nn_train = x[0:3830]
y_nn_train = y[0:3830]
y_nn_train = to_categorical(y_nn_train, num_classes=7)

x_nn_test = x[3830:]
y_nn_test = y[3830:]
y_nn_test = to_categorical(y_nn_test, num_classes=7)
print(len(x_nn_train), len(x_nn_test))
# создание и тренировка нейронки. эксперименты показали, что именно такая ахитектура нейронки яввляется наилучшей
model = Sequential()
model.add(LSTM(13, input_shape=(69,1,), return_sequences=1)) # добавляем слой LSTM, совместимый с Cuda при поддержке GPU
model.add(Dense(300, activation='relu')) #добавляем полносвязные слои
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.3)) # добавляем слой регуляризации, "выключая" указанное количество нейронов, во избежание переобучения
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5)) # добавляем слой регуляризации, "выключая" указанное количество нейронов, во избежание переобучения
#model.add(BatchNormalization()) # добавляем слой нормализации данных
model.add(Flatten()) # добавляем слой выравнивания/сглаживания ("сплющиваем" данные в вектор)
model.add(Dense(7, activation='softmax')) # добавляем полносвязный слой на 6 нейронов, с функцией активации softmax на выходном слое

model_checkpoint_c = ModelCheckpoint(filepath="./LSTM_Dense_best.h5", save_best_only=True, monitor="val_recall",
                                     verbose=1, mode="max")
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=[Recall(name="recall")])

with device("/cpu:0"):
    history = model.fit(x_nn_train, y_nn_train,
                    epochs=60,
                    verbose=1,
                    batch_size=4,
                    callbacks=[model_checkpoint_c],
                    validation_data = (x_nn_test, y_nn_test)
                   )

#считывание и сортировка тестового датасета
test_df=pd.read_csv("./test_dataset_test.csv")
test_df=test_df.sort_index(axis=1)
#выгрузка предсказаний на тестовом датасете
model  = load_model("./LSTM_Dense_best.h5")

test_df_nn = test_df.drop([".geo", "area"], axis = 1)
print(len(test_df_nn))



x = []


for f in range(len(test_df_nn)):
    x+=  [[test_df_nn.iloc[f].to_numpy()[0], test_df_nn.iloc[f].to_numpy()[1:70]]]
x = np.array(x)

#print(x)
res_nn_list = []

for f in x:
    #print(f)
    res_nn_list += [[f[0], predict_crop(f[1].reshape((1,69)))]]

outputNN = pd.DataFrame(data=res_nn_list, columns=["id", 'crop'])
outputNN.to_csv("./outputNN.csv", index=False)

#Выводим графики
plt.figure(figsize=(14,7))
plt.plot(history.history['recall'],
         label='Точность на обучающем наборе')
plt.plot(history.history['val_recall'],
         label='Точность на проверочном наборе')
plt.ylabel('Точность')
plt.legend()
plt.show()

plt.figure(figsize=(14,7))
plt.plot(history.history['loss'],
         label='Средняя абсолютная ошибка на обучающем наборе')
plt.plot(history.history['val_loss'],
         label='Средняя абсолютная ошибка на проверочном наборе')
plt.ylabel('Средняя ошибка')
plt.legend()
plt.show()
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=[Recall(name="recall")])

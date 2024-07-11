import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
# from datetime import datetime, timezone

# Local path to where the data is saved:
filename = 'C:/Users/Val/Desktop/Coding Projects/Data/eth.csv'

# Alternatively data can be easily downloaded:
# import yfinance as yf
# eth = yf.download(['eth-usd'], period = '2y', interval='1h')

# In this example I am defining 4 well known candlesticks patterns; more could be added later
class Signal(object):
    def __init__(self, filename, detect_ls, save_plot):
        self.data = pd.read_csv(filename)
        self.data["realbody"] = self.data.Close - self.data.Open
        self.time_period = '1H'
        self.detect_ls = detect_ls
        self.time_period = None
        self.save_plot = save_plot

    def trend(self, series):
        y = series.values.reshape(-1,1)
        x = np.array(range(1, series.shape[0] + 1)).reshape(-1,1)
        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_
        if slope > 0:
            return 1
        elif slope == 0:
            return 0
        else:
            return -1
        
    def dataframe_roll_evening(self, df):       
        def EveningStarSignal(window_series):            
            window_df = df.loc[window_series.index]           
            trend = window_df['trend1'].iloc[-4]
            body1 = window_df['realbody'].iloc[-3]
            body2 = window_df['realbody'].iloc[-2]
            body3 = window_df['realbody'].iloc[-1]
            half1 = window_df['Open'].iloc[-3] + (body1 * (1 / 2))
            half2 = window_df['Open'].iloc[-3] + (body1 * (4 / 5))
            close3 = window_df['Close'].iloc[-1]
            open2 = window_df['Open'].iloc[-2]                              
            cond1 = (trend == 1) and (body1 > 0) and (body2 > 0) and (body3 < 0)
            cond2 = (open2 > half1)
            cond3 = (close3 < half2)            
            if cond1 and cond2 and cond3:
                return 1  
            else:
                return 0            
        return EveningStarSignal
    
    def dataframe_roll_morning(self, df):    
        def MorningStarSignal(window_series):           
            window_df = df.loc[window_series.index]           
            trend = window_df['trend1'].iloc[-4]
            body1 = window_df['realbody'].iloc[-3]
            body2 = window_df['realbody'].iloc[-2]
            body3 = window_df['realbody'].iloc[-1]
            half1 = window_df['Open'].iloc[-3] + (body1 * (4 / 5))
            half2 = window_df['Open'].iloc[-3] + (body1 * (1 / 2))
            close3 = window_df['Close'].iloc[-1]
            open2 = window_df['Open'].iloc[-2]
            #percentile1 = stats.percentileofscore(abs(window_df['realbody']), abs(window_df['realbody'].iloc[-3]), kind='strict')            
            cond1 = (trend == -1) and (body1 < 0) and (body2 > 0) and (body3 > 0)
            cond2 = (close3 >= half1)
            cond3 = (open2 <= half2)
            #cond4 = (percentile1 > 60)                         
            if cond1 and cond2 and cond3:
                return 1 
            else:
                return 0          
        return MorningStarSignal
    
    def dataframe_roll_bear(self, df):   
        def BearishHaramiSignal(window_series):       
            window_df = df.loc[window_series.index]       
            trend = window_df['trend2'].iloc[-3]
            body1 = window_df['realbody'].iloc[-2]
            body2 = window_df['realbody'].iloc[-1]
            open1 = window_df['Open'].iloc[-2]
            open2 = window_df['Open'].iloc[-1]
            close1 = window_df['Close'].iloc[-2]
            close2 = window_df['Close'].iloc[-1]           
            cond1 = (trend == 1) and (body1 > 0) and (body2 < 0)
            cond2 = (open2 < close1)
            cond3 = (close2 > open1)          
            if cond1 and cond2 and cond3:
                return 1 
            else:
                return 0        
        return BearishHaramiSignal
    
    def dataframe_roll_bull(self, df):   
        def BullishHaramiSignal(window_series):       
            window_df = df.loc[window_series.index]       
            trend = window_df['trend2'].iloc[-3]
            body1 = window_df['realbody'].iloc[-2]
            body2 = window_df['realbody'].iloc[-1]
            open1 = window_df['Open'].iloc[-2]
            open2 = window_df['Open'].iloc[-1]
            close1 = window_df['Close'].iloc[-2]
            close2 = window_df['Close'].iloc[-1]             
            cond1 = (trend == -1) and (body1 < 0) and (body2 > 0)
            cond2 = (open2 > close1)
            cond3 = (close2 < open1)            
            if cond1 and cond2 and cond3:
                return 1 
            else:
                return 0        
        return BullishHaramiSignal
        
        
    def process(self):
        #self.data['Datetime'] = pd.to_datetime(self.data['Datetime'], format="%Y-%m-%d %H:%M:%S%z")
        #self.data.set_index('Datetime', inplace=True)
        #if (self.data.index[1] - self.data.index[0]).seconds == 60:
        #    self.time_period = '1m'
        #elif (self.data.index[1] - self.data.index[0]).seconds == 1800:
        #    self.time_period = '30m'
        #elif (self.data.index[1] - self.data.index[0]).seconds == 3600:
        #    self.time_period = '1H'
        #elif (self.data.index[1] - self.data.index[0]).days == 1:
        #    self.time_period = '1D'
        #elif (self.data.index[1] - self.data.index[0]).days == 7:
        #    self.time_period = '1W'
        self.data['trend1'] = self.data['Close'].rolling(7).apply(self.trend, raw=False)
        self.data['trend2'] = self.data['Close'].rolling(8).apply(self.trend, raw=False)
        
    def detect_all(self):
        for signal in self.detect_ls:
            if signal == 'MorningStar':
                self.data['MorningStar'] = self.data['Close'].rolling(4).apply(self.dataframe_roll_morning(self.data), raw=False)
                if self.save_plot == True: 
                    self.pattern(self.data, self.time_period, signal)
            
            elif signal == 'EveningStar':
                self.data['EveningStar'] = self.data['Close'].rolling(4).apply(self.dataframe_roll_evening(self.data), raw=False)
                if self.save_plot == True:
                    self.pattern(self.data, self.time_period, signal)
            
            elif signal == 'BearishHarami':
                self.data['BearishHarami'] = self.data['Close'].rolling(3).apply(self.dataframe_roll_bear(self.data), raw=False)
                if self.save_plot == True:
                    self.pattern(self.data, self.time_period, signal)
                
            elif signal == 'BullishHarami':
                self.data['BullishHarami'] = self.data['Close'].rolling(3).apply(self.dataframe_roll_bull(self.data), raw=False)
                if self.save_plot == True:
                    self.pattern(self.data, self.time_period, signal)
                    
        # file_name = 'C:/Users/Val/Desktop/Coding Projects/Data/btc_' + self.time_period + '_pattern.csv'
        file_name = 'C:/Users/Val/Desktop/Coding Projects/Data/btc_' + '1H' + '_pattern.csv'
        self.data.to_csv(file_name, index=False)
        return file_name
        
    def summary(self):
        period = self.data.index[[1, -1]]
        print('Rule : %s' % (self.time_period))
        print('Period : %s - %s' % (period[0], period[1]), '\n')
        total = self.data.shape[0]
        num = None
        for i in self.detect_ls:
            num = np.sum(self.data[i])
            print('Number of', i, ': %s // %s' % (num, total), '\n')
        

target = 'eth'
rule = '1H'
signal_ls = ['MorningStar', 'EveningStar', 'BearishHarami', 'BullishHarami']
save_plot = False

Sig = Signal(filename, signal_ls, save_plot)
Sig.process()
data_pattern = Sig.detect_all()
Sig.summary()


# creating the gramian angular field aproach used to transform time series data into images:

def ts2gasf(ts, max_v, min_v):
    '''
    Args:
        ts (numpy): (N, )
        max_v (int): max value for normalization
        min_v (int): min value for normalization

    Returns:
        gaf_m (numpy): (N, N)
    '''
    # Normalization : 0 ~ 1
    if max_v == min_v:
        gaf_m = np.zeros((len(ts), len(ts)))
    else:
        ts_nor = np.array((ts-min_v) / (max_v-min_v))
        # Arccos
        ts_nor_arc = np.arccos(ts_nor)
        # GAF
        gaf_m = np.zeros((len(ts_nor), len(ts_nor)))
        for r in range(len(ts_nor)):
            for c in range(len(ts_nor)):
                gaf_m[r, c] = np.cos(ts_nor_arc[r] + ts_nor_arc[c])
    return gaf_m

def get_gasf(arr):
    '''Convert time-series to gasf    
    Args:
        arr (numpy): (N, ts_n, 4)

    Returns:
        gasf (numpy): (N, ts_n, ts_n, 4)

    Todos:
        add normalization together version
    '''
    arr = arr.copy()
    gasf = np.zeros((arr.shape[0], arr.shape[1], arr.shape[1], arr.shape[2]))
    for i in range(arr.shape[0]):
        for c in range(arr.shape[2]):
            each_channel = arr[i, :, c]
            c_max = np.amax(each_channel)
            c_min = np.amin(each_channel)
            each_gasf = ts2gasf(each_channel, max_v=c_max, min_v=c_min)
            gasf[i, :, :, c] = each_gasf
    return gasf

def get_arr(data, signal, d=None):
    if signal != 'n':
        df_es = data.loc[data[signal]==1]
    else:
        df_es = d
    arr = np.zeros((df_es.shape[0], 10, 4))
    for index, N in zip(df_es.index, range(df_es.shape[0])):
        df = data.loc[data.index <= index][-10::]
        arr[N, :, 0] = df['Open']
        arr[N, :, 1] = df['High']
        arr[N, :, 2] = df['Low']
        arr[N, :, 3] = df['Close']
    return arr
       
def process(file):
    data = pd.read_csv(file)
    data['Datetime'] = pd.to_datetime(data['Datetime'], format="%Y-%m-%d %H:%M:%S%z")
    data.set_index('Datetime', inplace=True)
    return data

def detect(data, signal, d=None):
    arr = get_arr(data, signal, d)
    gasf = get_gasf(arr)
    return gasf

data_1D_pattern = pd.read_csv(data_pattern)
gasf_arr = np.zeros((len(signal_ls) + 1, 30, 10, 10, 4))
for i, j in zip(signal_ls, range(len(signal_ls))):
    gasf = detect(data_1D_pattern, i)
    gasf_arr[j, :, :, :, :] = gasf[0:30, :, :, :]
df = data_1D_pattern.copy()
for i in signal_ls: df = df.loc[df[i] != 1]
df = shuffle(df[9::])
gasf = detect(data_1D_pattern, 'n', df)
gasf_arr[-1, :, :, :, :] = gasf[0:30, :, :, :]

# Searialize the gasf_arr object to a binary file to be stored on disk:
#with open(self.gasf_arr, 'wb') as handle:    
#    pickle.dump(gasf_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Load the gasf_arr from disk:
#with open(self.gasf_arr, 'rb') as handle:
#    gasf_arr = pickle.load(handle)

x_arr = np.zeros(((len(signal_ls) + 1), 30, 10, 10, 4))
for i in range(len(signal_ls) + 1):
    x_arr[i, :, :, :, :] = gasf_arr[i, 0:30, :, :, :]    
x_arr = gasf_arr.reshape((len(signal_ls) + 1) * 30, 10, 10, 4)
y_arr = []
for i in range(len(signal_ls) + 1):
    ls = [i] * 30
    y_arr.extend(ls)
y_arr = np.array(y_arr)
load_data = {'data': x_arr, 'target': y_arr}
#self.load_data = 'load_data_' + self.target
#with open(self.load_data, 'wb') as handle:
#    pickle.dump(load_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Defining and fiting the neural network model using keras:
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers

class CNN(object):
    def __init__(self, arr_x, arr_y):
        self.arr_x, self.arr_y = arr_x, arr_y
        self.X_train_image, self.y_train_label, self.X_test_image, self.y_test_label = None, None, None, None
        self. y_trainOneHot, self.y_testOneHot = None, None
        self.input_shape = None
        self.label_shape = None
        self.model = None
        self.train_history = None
        
    def process(self):
        self.X_train_image, self.X_test_image, self.y_train_label, self.y_test_label = train_test_split (self.arr_x, self.arr_y, test_size= 0.3, random_state = 42)
        self. y_trainOneHot, self.y_testOneHot = to_categorical(self.y_train_label), to_categorical(self.y_test_label)
        self.input_shape = self.X_train_image[0].shape
        self.label_shape = self. y_trainOneHot.shape[1]
        
    def plot_image(self, image):
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        plt.imshow(image, cmap = 'binary')
        plt.show()
      
    def plot_images_labels_prediction(self, images, labels, prediction, idx, num=10):
        fig = plt.gcf()
        fig.set_size_inches(12, 14)
        if num > 25:
            num = 25
        for i in range(0, num):
            ax = plt.subplot(5, 5, 1 + i)
            ax.imshow(images[idx], cmap='binary')
            title = 'label = ' + str(labels[idx])
            if len(prediction) > 0:
                title += ', predict = ' + str(prediction[idx])
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
        plt.show()
    
    def show_train_history(self, train_history, train, validation):
        plt.figure()
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    
    def build(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=32,
                         kernel_size=(2, 2),
                         padding='same',
                         input_shape = self.input_shape,
                         activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=32,
                         kernel_size=(2, 2),
                         padding='same',
                         activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.model.add(Dropout(0.30))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(self.label_shape, activation='softmax'))           
        print(self.model.summary())
        
    def train(self, split):
        adam = optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.train_history = self.model.fit(x = self.X_train_image,
                                       y = self.y_trainOneHot,
                                       validation_split = split,
                                       epochs = 200,
                                       batch_size = 336,
                                       verbose = 2)
        
    def show(self):
        self.show_train_history(self.train_history, 'loss', 'val_loss')
        self.show_train_history(self.train_history, 'accuracy', 'val_accuracy')
        score = self.model.evaluate(self.X_test_image, self.y_testOneHot)
        print('Score of the Testing Data: {}'.format(score))
        pred = self.model.predict(self.X_test_image)
        prediction = np.argmax(pred, axis=1)
        self.plot_images_labels_prediction(self.X_test_image, self.y_test_label, prediction, idx=0)
        print(pd.crosstab(self.y_test_label, prediction, rownames=['label'], colnames=['predict']))
               
    def save(self, filename):
        self.model.save(filename)


#with open(self.load_data, 'rb') as handle:
#    load_data = pickle.load(handle)
x_arr = load_data['data']
y_arr = load_data['target']
model = CNN(x_arr, y_arr)
model.process()
model.build()
model.train(0.25)
model.show()
#self.load_model = 'CNN_' + self.target + '.h5'
#model.save(self.load_model)


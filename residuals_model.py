import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn as sk



import venv.Osah.tensorflow_docs as tfdocs
import venv.Osah.tensorflow_docs.plots
import venv.Osah.tensorflow_docs.modeling


class prepare_predict_dataset():
    def pred_dataset(self, data):
        dt_sample = pd.read_csv(data)
        dt = dt_sample.describe()
        dt = dt.transpose()
        dt_data =(dt_sample - dt['mean']) / dt['std']

        return dt_data


class model():
    def __init__(self, df):
        df.isna().sum()
        df.dropna()
        df.describe()
        train_samples = df.iloc[:, 0:int(len(df.columns) - 1)]
        target_samples = df.iloc[:, -1]
        self.dt = df


        # #Training dataset
        self.train_dataset = train_samples.sample(frac=0.8)
        self.test_dataset = train_samples.drop(self.train_dataset.index)
        print(self.train_dataset)
        #
        # #target dataset
        self.train_labels = target_samples.sample(frac=0.8)
        self.test_labels = target_samples.drop(self.train_labels.index)
        print(self.train_labels)




    def display_plot(self, labels):
        sns.pairplot(self.dt[labels], diag_kind="kde")
        plt.show()



    def data_description(self, desc):

        self.train_stats = self.train_dataset.describe()
        self.train_stats = self.train_stats.transpose()
        print("Training Stats: ", self.train_stats)


    def norm(self,x):
        return (x - self.train_stats['mean']) / self.train_stats['std']

    def build_model(self, activation_func='relu'):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(200, activation=activation_func, input_shape=(self.train_dataset.shape[1],)),
            tf.keras.layers.Dense(200, activation=activation_func),
            tf.keras.layers.Dense(128, activation=activation_func),
            tf.keras.layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)



        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mse', 'mae']
                     )
        return model


    def train(self,act_fun='relu',epochs=300,pred_data=None):
        model = self.build_model(activation_func=act_fun)
        ppd = prepare_predict_dataset()
        pd = ppd.pred_dataset(data=pred_data)
        print("pd: ", pd)
        self.norm_trained_data = self.norm(self.train_dataset)
        self.norm_test_data = self.norm(self.test_dataset)
      #
        self.history = model.fit(
            self.norm_trained_data, self.train_labels,epochs=epochs,
            validation_split=0.2, verbose=0)
        #print(self.history)
        hist = self.checkHistory()
      #  self.plotter()
        self.tran2(epochs=epochs,model=model)
        eval = self.evaluate(model=model)
        pred = self.predict(model=model,pred_data=pd)
        #self.predict2(model=model)
        return hist, eval, pred

    def tran2(self,epochs,model):
        plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        early_history = model.fit(self.norm_trained_data, self.train_labels,
                                  epochs=epochs, validation_split=0.2, verbose=0,
                                  callbacks=[early_stop, tfdocs.modeling.EpochDots()])
        plotter.plot({'Early Stopping': early_history}, metric="mae")
        plt.ylim([0, 10])
        plt.ylabel('MAE')

    def checkHistory(self):
        hist = pd.DataFrame(self.history.history)
        hist['epoch'] = self.history.epoch
        hist.tail()
        return hist

    def plotter(self):
        plot = tfdocs.plots.HistoryPlotter(smoothing_std=2)
        #MAE
        plot.plot({"" : self.history}, metric="mae")
        plt.ylim([-0.01, 0.05])
        plt.ylabel('MAE')
        plt.show()
        #MSE
        plot.plot({"" :  self.history}, metric="mse")
        plt.ylim([-0.01, 0.05])
        plt.ylabel('MSE')
        plt.show()


    def evaluate(self,model):
        loss, mae, mse = model.evaluate(self.norm_test_data, self.test_labels, verbose=2)
       # rmse = np.sqrt(mse)

        return loss,mse,mae

    def predict2(self,model):

        test_predictions = model.predict(self.norm_test_data)
        print("test prediction: ", test_predictions)


    def predict(self, model, pred_data=None):
        predictions = model.predict(pred_data)
        print("test prediction: ", predictions)
        return predictions
        # plt.axes(aspect='equal')
        # plt.scatter(self.test_labels, test_predictions)
        # plt.plot()
        # plt.xlabel('True Values')
        # plt.ylabel('Predictions')
        # # lims = [-0.01, 0.5]
        # plt.xlim(lims)
        # plt.ylim(lims)
        # _ = plt.plot(-0.01,0.9)
        # plt.show()

        # error = test_predictions - self.test_labels
        # print("Error: ",error)
        # plt.hist(error, bins=30)
        # plt.xlabel("Prediction Error")
        # _ = plt.ylabel("Count")
        #
        # plt.show()
        # return test_predictions,error



    def saveModel(self,path):
        self.build_model(activation_func='relu').save(path)


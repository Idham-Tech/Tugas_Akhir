import streamlit as st
from plotly import graph_objs as go 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from statsmodels.tsa.seasonal import STL
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, GRU, Dense, Dropout, Input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import load_model

def plot_actual_data(dates, data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=data, name='NonMigas'))
    fig.layout.update(title_text='Non-Oil and Gas', xaxis_rangeslider_visible=True, hovermode = 'x')
    st.plotly_chart(fig)

def plot_train(dates_train, train_act, train_pred):
    figgrutrain = go.Figure()
    train_act = train_act.flatten()
    train_pred = train_pred.flatten()
    figgrutrain.layout.update(title_text=('Actual and Predicted With GRU MODEL (Train)'), xaxis_rangeslider_visible=True, hovermode = 'x')
    figgrutrain.add_trace(go.Scatter(x=dates_train, y=train_act, name='Actual Value'))
    figgrutrain.add_trace(go.Scatter(x=dates_train, y=train_pred, name='Predicted Low Price'))
    st.plotly_chart(figgrutrain)

def plot_predict(dates_test, test_act, test_pred):
    figgrutest = go.Figure()
    test_act = test_act.flatten()
    test_pred = test_pred.flatten()
    figgrutest.layout.update(title_text=('Actual and Predicted With GRU MODEL (Test)'), xaxis_rangeslider_visible=True, hovermode = 'x')
    figgrutest.add_trace(go.Scatter(x=dates_test, y=test_act, name='Actual Value'))
    figgrutest.add_trace(go.Scatter(x=dates_test, y=test_pred, name='Predicted Low Price'))
    st.plotly_chart(figgrutest)

def stl_plot_train(dates_train, act, train):
    figgrutrain = go.Figure()
    date_train = dates_train.index
    train_act = act
    train_pred = train[:]
    figgrutrain.layout.update(title_text=('Actual and Predicted With GRU MODEL (Train)'), xaxis_rangeslider_visible=True, hovermode = 'x')
    figgrutrain.add_trace(go.Scatter(x=date_train, y=train_act, name='Actual Value'))
    figgrutrain.add_trace(go.Scatter(x=date_train, y=train_pred, name='Predicted Low Price'))
    st.plotly_chart(figgrutrain)

def stl_plot_predict(dates_test, act, test):
    figgrutest = go.Figure()
    date_test = dates_test.index
    test_act = act
    test_pred = test[:]
    figgrutest.layout.update(title_text=('Actual and Predicted With GRU MODEL (Test)'), xaxis_rangeslider_visible=True, hovermode = 'x')
    figgrutest.add_trace(go.Scatter(x=date_test, y=test_act, name='Actual Value'))
    figgrutest.add_trace(go.Scatter(x=date_test, y=test_pred, name='Predicted Low Price'))
    st.plotly_chart(figgrutest)

def comp_plot(df):
    st.bar_chart(df,x='Model', y='MAE', stack=False)
    st.bar_chart(df,x='Model', y='RMSE', stack=False)
    st.bar_chart(df,x='Model', y='MAPE', stack=False)

def forcast(model, data_scaled, time_steps, scaler):
    last_sequence = data_scaled[-time_steps:].reshape((1, time_steps, 1))
    next_month_prediction = model.predict(last_sequence)
    next_month_prediction = scaler.inverse_transform(next_month_prediction)
    return next_month_prediction
    
def forcast_stl(model, data, time_steps, scaler):
    stl_last = STL(data['NonMigas'], seasonal=13)
    result_last = stl_last.fit()

    # Extract the last trend and seasonal components
    last_trend = result_last.trend[-1]
    last_seasonal = result_last.seasonal[-1]

    # Step 2: Use the GRU model to predict the residual for the next month
    # We use the last few residuals from the training set to predict the next one
    residuals = result_last.resid
    residuals_scaled = scaler.transform(residuals.values.reshape(-1, 1))
    last_residuals = residuals_scaled[-time_steps:]

    X_next = last_residuals.reshape(1, time_steps, 1)


    # Predict the residual for the next month
    predicted_residual_next = model.predict(X_next)
    predicted_residual_next = scaler.inverse_transform(predicted_residual_next).flatten()[0]

    # Step 3: Combine the predicted residual with the last known trend and seasonal components
    next_month = last_trend + last_seasonal + predicted_residual_next
    return next_month

def evaluation(y_test_inv, test_predictions):
    test_mae = mean_absolute_error(y_test_inv, test_predictions)
    test_mape = mean_absolute_percentage_error(y_test_inv, test_predictions) * 100
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predictions))
    test_r2 = r2_score(y_test_inv, test_predictions)
    st.write('MAE:', test_mae, '  \nRMSE:', test_rmse, '  \nMAPE:', test_mape, '  \nR^2:', test_r2)
    return test_mae, test_rmse, test_mape, test_r2

def stk_gru_models():
    # Preprocess the data
    data = pd.read_csv('./Keseluruhan NonMigas.csv', parse_dates=['date'], index_col='date')
    
    # Extract 'NonMigas' and normalize
    data_unscaled = data['NonMigas'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_unscaled)

    # Split data into training and testing sets
    train_size = int(len(data_scaled) * 0.75)
    train, test = data_scaled[:train_size], data_scaled[train_size:]

    # Create sequences for training the GRU model
    def create_sequences(data, time_steps):
        sequences = []
        labels = []
        for i in range(len(data) - time_steps):
            sequences.append(data[i:i + time_steps])
            labels.append(data[i + time_steps])
        return np.array(sequences), np.array(labels)

    time_steps = 3  

    X_train, y_train = create_sequences(train, time_steps)
    X_test, y_test = create_sequences(test, time_steps)

    def build_Stacked_GRU(input_shape):
        model = models.Sequential()
        model.add(GRU(64, return_sequences=True,kernel_regularizer=l2(0.01), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(64, return_sequences=True))
        model.add(GRU(64))
        model.add(Dense(1))
        adam = Adam(learning_rate=0.001)
        model.compile(optimizer=adam, loss='mean_squared_error')
        return model

    # Set up input shape
    input_shape = (time_steps, 1)

    # Build the model
    stk_model = build_Stacked_GRU(input_shape)

    history = stk_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1, validation_split=0.2)
    # Make predictions on the training set
    train_predictions = stk_model.predict(X_train)
    # Make predictions on the test set
    test_predictions = stk_model.predict(X_test)

    # Inverse transform the training predictions and actual values
    stk_train_predictions_inv = scaler.inverse_transform(train_predictions)
    stk_y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))

    # Inverse transform the test predictions and actual values
    stk_test_predictions_inv = scaler.inverse_transform(test_predictions)
    stk_y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    next_month = forcast(stk_model, data_scaled, time_steps, scaler)
    stk_next_month = np.float64(next_month)

    return stk_test_predictions_inv, stk_y_test_inv, stk_train_predictions_inv, stk_y_train_inv, data, stk_next_month

def bid_gru_models():
    # Preprocess the data
    data = pd.read_csv('./Keseluruhan NonMigas.csv', parse_dates=['date'], index_col='date')
    
    # Extract 'NonMigas' and normalize
    data_unscaled = data['NonMigas'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_unscaled)

    # Split data into training and testing sets
    train_size = int(len(data_scaled) * 0.75)
    train, test = data_scaled[:train_size], data_scaled[train_size:]

    # Create sequences for training the GRU model
    def create_sequences(data, time_steps):
        sequences = []
        labels = []
        for i in range(len(data) - time_steps):
            sequences.append(data[i:i + time_steps])
            labels.append(data[i + time_steps])
        return np.array(sequences), np.array(labels)

    time_steps = 3  

    X_train, y_train = create_sequences(train, time_steps)
    X_test, y_test = create_sequences(test, time_steps)

    def build_bidirectional_gru(input_shape):
        model = models.Sequential()
        model.add(layers.Bidirectional(layers.GRU(64, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=input_shape))
        model.add(Dropout(0.1))
        model.add(layers.Bidirectional(layers.GRU(64)))
        model.add(layers.Dense(1))
        adam = Adam(learning_rate=0.001)
        model.compile(optimizer=adam, loss='mse')
        return model

    # Set up input shape
    input_shape = (time_steps, 1)

    # Build the model
    bid_model = build_bidirectional_gru(input_shape)

    history = bid_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))    # Make predictions on the training set

    # Make predictions
    train_predictions = bid_model.predict(X_train)
    test_predictions = bid_model.predict(X_test)

    # Rescale predictions back to original values
    bid_test_predictions_inv = scaler.inverse_transform(test_predictions)
    bid_y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Rescale predictions back to original values
    bid_train_predictions_inv = scaler.inverse_transform(train_predictions)
    bid_y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))

    next_month = forcast(bid_model, data_scaled, time_steps, scaler)
    bid_next_month = np.float64(next_month)

    return bid_test_predictions_inv, bid_y_test_inv, bid_train_predictions_inv, bid_y_train_inv, data, bid_next_month

def att_gru_models():
    # Preprocess the data
    data = pd.read_csv('./Keseluruhan NonMigas.csv', parse_dates=['date'], index_col='date')
    
    # Extract 'NonMigas' and normalize
    data_unscaled = data['NonMigas'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_unscaled)

    # Split data into training and testing sets
    train_size = int(len(data_scaled) * 0.75)
    train, test = data_scaled[:train_size], data_scaled[train_size:]

    # Create sequences for training the GRU model
    def create_sequences(data, time_steps):
        sequences = []
        labels = []
        for i in range(len(data) - time_steps):
            sequences.append(data[i:i + time_steps])
            labels.append(data[i + time_steps])
        return np.array(sequences), np.array(labels)

    time_steps = 3  

    X_train, y_train = create_sequences(train, time_steps)
    X_test, y_test = create_sequences(test, time_steps)

    class AttentionLayer(Layer):
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="normal")
            self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros")
            self.U = self.add_weight(name="att_u", shape=(input_shape[-1],), initializer="normal")
            super(AttentionLayer, self).build(input_shape)

        def call(self, x):
            # Compute attention scores
            u_score = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
            u_weight = tf.tensordot(u_score, self.U, axes=1)
            att_weight = tf.nn.softmax(u_weight, axis=1)

            # Weighted sum of inputs according to attention scores
            output = x * tf.expand_dims(att_weight, -1)
            return tf.reduce_sum(output, axis=1)

    def build_attention_gru_model(time_steps, features):
        inputs = Input(shape=(time_steps, features))

        # 1st GRU layer with return sequences and L2 regularization
        gru_output = GRU(64, return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
        dropout_1 = Dropout(0.3)(gru_output)  # Dropout layer

        # 2nd GRU layer with return sequences and L2 regularization
        gru_output_2 = GRU(64, return_sequences=True)(dropout_1)
        # Dropout layer

        # 3rd GRU layer with return sequences
        gru_output_3 = GRU(64, return_sequences=True)(gru_output_2)
        # Dropout layer

        # Attention Layer
        attention_output = AttentionLayer()(gru_output_3)

        # Output layer (for regression)
        output = Dense(1)(attention_output)

        # Define the model
        model = Model(inputs=inputs, outputs=output)

        # Compile the model
        adam = Adam(learning_rate=0.001)
        model.compile(optimizer=adam, loss='mean_squared_error')
        return model


    # Build the model
    model_attention_gru = build_attention_gru_model(time_steps=time_steps, features=1)

    history = model_attention_gru.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))

    # Make predictions
    test_predictions_attention_gru = model_attention_gru.predict(X_test)
    train_predictions_attention_gru = model_attention_gru.predict(X_train)

    # Inverse transform predictions ke skala asli
    test_predictions_attention_gru_inv = scaler.inverse_transform(test_predictions_attention_gru)
    train_predictions_attention_gru_inv = scaler.inverse_transform(train_predictions_attention_gru)

    # Inverse transform actual Y_test values
    att_Y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Inverse transform actual Y_test values
    att_Y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))

    next_month = forcast(model_attention_gru, data_scaled, time_steps, scaler)
    att_next_month = np.float64(next_month)

    return test_predictions_attention_gru_inv, att_Y_test_inv, train_predictions_attention_gru_inv, att_Y_train_inv, data, att_next_month

def stl_gru_models():
    # stl_model = model
    # Preprocess the data
    data = pd.read_csv('./Keseluruhan NonMigas.csv', parse_dates=['date'], index_col='date')
    
    # Split data into training and testing sets (75% train, 25% test)
    train_size = int(len(data) * 0.75)
    train_data, test_data = data[:train_size], data[train_size:]

    # STL decomposition on training set
    stl = STL(train_data['NonMigas'], seasonal=13)
    result = stl.fit()

    # Extract trend, seasonal, and residual components from training data
    trend_train = result.trend
    seasonal_train = result.seasonal
    residual_train = result.resid

    # Normalize residuals for GRU model
    scaler = MinMaxScaler()
    residual_train_scaled = scaler.fit_transform(residual_train.values.reshape(-1, 1))

    # Create sequences for training the GRU model
    def create_sequences(data, time_steps):
        sequences = []
        labels = []
        for i in range(len(data) - time_steps):
            sequences.append(data[i:i + time_steps])
            labels.append(data[i + time_steps])
        return np.array(sequences), np.array(labels)

    time_steps = 3  
    X_train, y_train = create_sequences(residual_train_scaled, time_steps)

    # Reshape for GRU input (samples, time steps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    stl_model = Sequential()
    stl_model.add(GRU(128, return_sequences=True,kernel_regularizer=l2(0.01), input_shape=(time_steps, 1)))
    stl_model.add(Dropout(0.3))
    stl_model.add(GRU(128))
    stl_model.add(Dense(1))
    adam = Adam(learning_rate=0.001)
    stl_model.compile(optimizer=adam, loss='mean_squared_error')

    history = stl_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)
    
    # Perform STL decomposition on test set
    stl_test = STL(test_data['NonMigas'], seasonal=13)
    result_test = stl_test.fit()

    # Extract components from the test set
    trend_test = result_test.trend
    seasonal_test = result_test.seasonal
    residual_test = result_test.resid

    # Normalize test residuals
    residual_test_scaled = scaler.transform(residual_test.values.reshape(-1, 1))

    # Prepare test data for GRU prediction
    X_test, y_test = create_sequences(residual_test_scaled, time_steps)

    # Reshape test data
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Make predictions
    stl_predictions_test = stl_model.predict(X_test)
    stl_predictions_train = stl_model.predict(X_train)

    # Inverse transform the predictions to get the original scale
    stl_predictions_test = scaler.inverse_transform(stl_predictions_test).flatten() # Flatten the array
    stl_predictions_train = scaler.inverse_transform(stl_predictions_train).flatten() # Flatten the array

    # Combine GRU predictions with trend and seasonal components from the test set
    final_predictions_test = trend_test[-len(stl_predictions_test):] + seasonal_test[-len(stl_predictions_test):] + stl_predictions_test

    # Combine GRU predictions with trend and seasonal components from the train set
    final_predictions_train = trend_train[-len(stl_predictions_train):] + seasonal_train[-len(stl_predictions_train):] + stl_predictions_train

    stl_y_train_inv = train_data['NonMigas'][-len(final_predictions_train):]
    stl_y_test_inv = test_data['NonMigas'][-len(final_predictions_test):]

    stl_next_month = forcast_stl(stl_model, data, time_steps, scaler)

    return final_predictions_test, stl_y_test_inv, stl_y_train_inv, final_predictions_train, data, stl_next_month


def data_act():
    st.subheader('Data Actual')
def visual_data():
    st.subheader('Visualizations Data')
def visual_trainval():
    st.subheader('Visualizations Train and Validation Loss')
def visual_actpred_datatrain():
    st.subheader('Visualizations Actual and Prediction Data Train')
def visual_actpred_data():
    st.subheader('Visualizations Actual and Prediction Data')
def write_evaluation():
    st.subheader('Evaluation')
def write_forecast():
    st.subheader('Forecasting')

def proccess(option):
    # load model
    if option == 'Stacked GRU':
        # model = tf.keras.models.load_model('./models/stk_modelgru.h5')
        stk_test_predictions_inv, stk_y_test_inv, stk_train_predictions_inv, stk_y_train_inv, data, stk_next_month= stk_gru_models()
            
        visual_actpred_data()
        data_size = int(len(data) * 0.75)
        date_train, date_test = data[:data_size], data[data_size:]
        plot_train(date_train.index, stk_y_train_inv, stk_train_predictions_inv)
        plot_predict(date_test.index, stk_y_test_inv, stk_test_predictions_inv)
            
    
        #evaluation
        write_evaluation()
        evaluation(stk_y_test_inv, stk_test_predictions_inv)
        
        #forcasting
        st.write('🙌🏻Next month export price forecast:', stk_next_month)

    elif option == 'Bidirectional GRU':
        # model = tf.keras.models.load_model('./models/bid_modelgru.h5')
        bid_test_predictions_inv, bid_y_test_inv, bid_train_predictions_inv, bid_y_train_inv, data, bid_next_month= bid_gru_models()
            
        visual_actpred_data()
        data_size = int(len(data) * 0.75)
        date_train, date_test = data[:data_size], data[data_size:]
        plot_train(date_train.index, bid_y_train_inv, bid_train_predictions_inv)
        plot_predict(date_test.index, bid_y_test_inv, bid_test_predictions_inv)
            
    
        #evaluation
        write_evaluation()
        evaluation(bid_y_test_inv, bid_test_predictions_inv)
        
        #forcasting
        st.write('🙌🏻Next month export price forecast:', bid_next_month)   

    elif option == 'Attention + GRU':
        # model = tf.keras.models.load_model('./models/att_modelgru.h5')
        test_predictions_attention_gru_inv, att_Y_test_inv, train_predictions_attention_gru_inv, att_Y_train_inv, data, att_next_month= att_gru_models()
            
        visual_actpred_data()
        data_size = int(len(data) * 0.75)
        date_train, date_test = data[:data_size], data[data_size:]
        plot_train(date_train.index, att_Y_train_inv, train_predictions_attention_gru_inv)
        plot_predict(date_test.index, att_Y_test_inv, test_predictions_attention_gru_inv)
            
    
        #evaluation
        write_evaluation()
        evaluation(att_Y_test_inv, test_predictions_attention_gru_inv)
        
        #forcasting
        st.write('🙌🏻Next month export price forecast:', att_next_month)     

    elif option == 'STL + GRU':
        # model = tf.keras.models.load_model('./models/stl_modelgru.h5')
        stl_final_predictions_test, stl_y_test_inv, stl_y_train_inv, stl_final_predictions_train, data, stl_next_month= stl_gru_models()
            
        visual_actpred_data()
        data_size = int(len(data) * 0.75)
        date_train, date_test = data[:data_size], data[data_size:]
        stl_plot_train(date_train, stl_y_train_inv, stl_final_predictions_train)
        stl_plot_predict(date_test, stl_y_test_inv, stl_final_predictions_test)
            
    
        #evaluation
        write_evaluation()
        evaluation(stl_y_test_inv, stl_final_predictions_test)
        
        #forcasting
        st.write('🙌🏻Next month export price forecast:', stl_next_month)       



    else:
        stk_test_predictions_inv, stk_y_test_inv, stk_train_predictions_inv, stk_y_train_inv, data, stk_next_month= stk_gru_models()
        bid_test_predictions_inv, bid_y_test_inv, bid_train_predictions_inv, bid_y_train_inv, data, bid_next_month= bid_gru_models()
        test_predictions_attention_gru_inv, att_Y_test_inv, train_predictions_attention_gru_inv, att_Y_train_inv, data, att_next_month= att_gru_models()
        stl_final_predictions_test, stl_y_test_inv, stl_y_train_inv, stl_final_predictions_train, data, stl_next_month= stl_gru_models()

        visual_actpred_data()
        data_size = int(len(data) * 0.75)
        date_train, date_test = data[:data_size], data[data_size:]
        
        st.write('Stacked GRU')
        plot_predict(date_test.index, stk_y_test_inv, stk_test_predictions_inv)
        
        st.write('Bidirectional GRU')
        plot_predict(date_test.index, bid_y_test_inv, bid_test_predictions_inv)
        
        st.write('Attention Based - GRU')
        plot_predict(date_test.index, att_Y_test_inv, test_predictions_attention_gru_inv)

        st.write('STL - GRU')
        stl_plot_predict(date_test, stl_y_test_inv, stl_final_predictions_test)

        #evaluation
        write_evaluation()
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            st.write(f"**{'Stacked GRU'}**")
            stk_mae,stk_rmse,stk_mape,stk_r2 = evaluation(stk_y_test_inv, stk_test_predictions_inv)
        with col5:
            st.write(f"**{'Bidirectional GRU'}**")
            bid_mae,bid_rmse,bid_mape, bid_r2 = evaluation(bid_y_test_inv,  bid_test_predictions_inv)
        with col6:
            st.write(f"**{'Attention Based - GRU'}**")
            att_mae,att_rmse,att_mape, att_r2 = evaluation(att_Y_test_inv,  test_predictions_attention_gru_inv)
        with col7:
            st.write(f"**{'STL - GRU'}**")
            stl_mae,stl_rmse,stl_mape, stl_r2 = evaluation(stl_y_test_inv,  stl_final_predictions_test)

        comp = {'Model': ['stk_gru', 'bid_gru', 'att_gru', 'stl_gru'],
        'MAE': [stk_mae, bid_mae, att_mae, stl_mae],
        'RMSE': [stk_rmse, bid_rmse, att_rmse, stl_rmse],
        'MAPE': [stk_mape, bid_mape, att_mape, stl_mape],
        'R^2' : [stk_r2, bid_r2, att_r2, stl_r2]
        }
        df_comp = pd.DataFrame(comp)
        
        comp_plot(df_comp)

        #forcasting
        write_forecast()
        col9, col10, col11, col12 = st.columns(4)
        with col9:
            st.write('Stacked GRU')
            st.write('🙌🏻Next month export price forecast:', stk_next_month)
        with col10:
            st.write('Bidirectional GRU')
            st.write('🙌🏻Next month export price forecast:', bid_next_month)
        with col11:
            st.write('Attention Based - GRU')
            st.write('🙌🏻Next month export price forecast:', att_next_month)  
        with col12:
            st.write('STL - GRU')
            st.write('🙌🏻Next month export price forecast:', stl_next_month)  
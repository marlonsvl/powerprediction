from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render, redirect
from django.urls import reverse
from django.views import generic
from .models import Choice, Question

from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .forms import NameForm

import datetime
import pandas as pd
import csv
import os
from django.conf import settings
import IPython
import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import backend as K
from django_matplotlib import MatplotlibFigureField
from .figures import get_plot, get_bar, get_determination, get_real_neural_normed, get_real_neural_kw
import json

class IndexView(generic.ListView):
    template_name = 'forecasting/index.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        """Return the last five published questions."""
        return Question.objects.order_by('-pub_date')[:5]


def home(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = NameForm(request.POST)
        
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            print("great is valid !!!!")
            path = settings.DATA_FOLDER+'UGE_01_Filter.csv'
            #file_ = open(os.path.join(settings.BASE_DIR, 'UGE_01_Filter.csv'))
            df = pd.read_csv(path)
            print(df.shape)
            ##############
            date_time = pd.to_datetime(df.pop('DATE'), format='%Y.%m.%d %H:%M:%S')

            timestamp_s = date_time.map(datetime.datetime.timestamp)


            day = 24*60*60
            year = (365.2425)*day

            df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
            df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
            df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
            df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

            """
            fft = tf.signal.rfft(df['POWER'])
            f_per_dataset = np.arange(0, len(fft))

            n_samples_h = len(df['POWER'])
            hours_per_year = 24*365.2524
            years_per_dataset = n_samples_h/(hours_per_year)

            f_per_year = f_per_dataset/years_per_dataset

            column_indices = {name: i for i, name in enumerate(df.columns)}
            """
            n = len(df)
            train_df = df[0:int(n*0.8)]
            val_df = df[int(n*0.8):int(n*0.95)]
            test_df = df[int(n*0.95):]

            num_features = df.shape[1]
            

            ########## ploting ##########
                    
            #train_df['POWER'].plot(legend=True,figsize=(16,8))
            chart = get_plot(train_df['POWER'])
            #val_df['POWER'].plot(legend=True)
            chart1 = get_plot(val_df['POWER'])
            #test_df['POWER'].plot(legend=True)
            chart2 = get_plot(test_df['POWER'])
            # Data for plotting
                  


            train_mean = train_df.mean()
            train_std = train_df.std()

            train_df = (train_df - train_mean) / train_std
            val_df = (val_df - train_mean) / train_std
            test_df = (test_df - train_mean) / train_std

            df_std = (df - train_mean) / train_std
            df_std = df_std.melt(var_name='Column', value_name='Normalized')
            

            class WindowGenerator():
              def __init__(self, input_width, label_width, shift,
                       train_df=train_df, val_df=val_df, test_df=test_df,
                       label_columns=None):
            # Store the raw data.
                self.train_df = train_df
                self.val_df = val_df
                self.test_df = test_df

            # Work out the label column indices.
                self.label_columns = label_columns
                if label_columns is not None:
                  self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
                self.column_indices = {name: i for i, name in
                                   enumerate(train_df.columns)}

            # Work out the window parameters.
                self.input_width = input_width
                self.label_width = label_width
                self.shift = shift

                self.total_window_size = input_width + int(shift)

                self.input_slice = slice(0, input_width)
                self.input_indices = np.arange(self.total_window_size)[self.input_slice]

                self.label_start = int(self.total_window_size - int(self.label_width))
                self.labels_slice = slice(self.label_start, None)
                self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

              def __repr__(self):
                return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])


            def split_window(self, features):
              inputs = features[:, self.input_slice, :]
              labels = features[:, self.labels_slice, :]
              if self.label_columns is not None:
                labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

          # Slicing doesn't preserve static shape information, so set the shapes
          # manually. This way the `tf.data.Datasets` are easier to inspect.
              inputs.set_shape([None, self.input_width, None])
              labels.set_shape([None, int(self.label_width), None])

              return inputs, labels

            WindowGenerator.split_window = split_window

            
            def plot(self, model=None, plot_col='POWER', max_subplots=3):
              pred = []
              inputs, labels = self.example
              #plt.figure(figsize=(12, 3))
              plot_col_index = self.column_indices[plot_col]
              max_n = min(max_subplots, len(inputs))
              for n in range(max_n):
                #plt.subplot(3, 1, n+1)
                #plt.ylabel(f'{plot_col} [normed]')
                #plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                #     label='Inputs', marker='.', zorder=-10)

                if self.label_columns:
                  label_col_index = self.label_columns_indices.get(plot_col, None)
                else:
                  label_col_index = plot_col_index

                if label_col_index is None:
                  continue
            
                trues = labels[n, :, label_col_index].numpy()
                
                #plt.scatter(self.label_indices, labels[n, :, label_col_index],
                #        edgecolors='k', label='Labels', c='#2ca02c', s=64)
                if model is not None:
                  predictions = model(inputs)
                  #plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  #        marker='X', edgecolors='k', label='Predictions',
                  #        c='#ff7f0e', s=64)
                  pred = predictions[n, :, label_col_index].numpy()  
                #if n == 0:
                  #plt.legend()

              #plt.xlabel('Time [h]')
              return trues, pred

            WindowGenerator.plot = plot
            #chart3 = plot()
            

            

            def make_dataset(self, data):
              data = np.array(data, dtype=np.float32)
              ds = tf.keras.preprocessing.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=self.total_window_size,
              sequence_stride=1,
              shuffle=True,
              batch_size=30,)

              ds = ds.map(self.split_window)

              return ds

            WindowGenerator.make_dataset = make_dataset

            @property
            def train(self):
              return self.make_dataset(self.train_df)

            @property
            def val(self):
              return self.make_dataset(self.val_df)

            @property
            def test(self):
              return self.make_dataset(self.test_df)

            @property
            def example(self):
              #Get and cache an example batch of `inputs, labels` for plotting
              result = getattr(self, '_example', None)
              if result is None:
                # No example batch was found, so get one from the `.train` dataset
                result = next(iter(self.train))
                # And cache it for next time
                self._example = result
              return result

            WindowGenerator.train = train
            WindowGenerator.val = val
            WindowGenerator.test = test
            WindowGenerator.example = example

            multi_window = WindowGenerator(input_width=48,
                                           label_width=form['out_steps'].value(),
                                           shift=form['out_steps'].value())

            ######## ploting ##########
            multi_window.plot()

            
            multi_val_performance = {}
            multi_performance = {}
            
            
            ## st.header("Hybrid Model") 
            def compile_and_fit(model, window, lr, patience=2):
              early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                patience=patience,
                                                                mode='min')
              
              def coeff_determination(y_true, y_pred):
                  SS_res =  K.sum(K.square( y_true-y_pred )) 
                  SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
                  return ( 1 - SS_res/(SS_tot + K.epsilon()) ) 
                
              
              model.compile(loss=tf.losses.MeanSquaredError(),
                            optimizer=tf.optimizers.SGD(lr=lr, momentum=0.95),
                            metrics=[tf.metrics.MeanAbsoluteError(), coeff_determination, 
                                           tf.keras.metrics.MeanAbsolutePercentageError(),
                                          tf.keras.metrics.MeanSquaredError()])

              history = model.fit(window.train, epochs=int(form['max_epochs'].value()),
                                  validation_data=window.val,
                                  callbacks=[early_stopping])
              return history


            
            multi_hybrid_model = tf.keras.Sequential([
                 # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
                tf.keras.layers.Lambda(lambda x: x[:, -int(form['conv_width'].value()):, :]),
                # Shape => [batch, 1, conv_units]
                tf.keras.layers.Conv1D(192, activation='relu', kernel_size=(int(form['conv_width'].value())),
                                      strides=1,
                                      input_shape=[None, 1]),

                tf.keras.layers.LSTM(32, return_sequences=True),
                tf.keras.layers.Dense(24, activation="relu"),
                tf.keras.layers.Dense(24, activation="relu"),
                
                tf.keras.layers.Dense(int(form['out_steps'].value())*num_features,
                                      kernel_initializer=tf.initializers.zeros),
                # Shape => [batch, out_steps, features]
                tf.keras.layers.Reshape([int(form['out_steps'].value()), num_features]),
            ])
            lr = 3e-2
            history = compile_and_fit(multi_hybrid_model, multi_window, lr)

            
            
            IPython.display.clear_output()

            multi_val_performance['Hybrid'] = multi_hybrid_model.evaluate(multi_window.val)
            multi_performance['Hybrid'] = multi_hybrid_model.evaluate(multi_window.test, verbose=0)
            trues, hybrid_pred = multi_window.plot(multi_hybrid_model)

            



            x = np.arange(len(multi_performance))
            width = 0.3
            metric_name = 'mean_absolute_error'
            metric_index = multi_hybrid_model.metrics_names.index('mean_absolute_error')
            val_mae = [v[metric_index] for v in multi_val_performance.values()]
            test_mae = [v[metric_index] for v in multi_performance.values()]

            #plt.bar(x - 0.17, val_mae, width, label='Validation')
            chart3 = get_bar(x - 0.17, val_mae, width, 'Validation', test_mae, 'Test', multi_performance)
            #plt.bar(x + 0.17, test_mae, width, label='Test')
            #plt.xticks(ticks=x, labels=multi_performance.keys(),
            #           rotation=45)
            #plt.ylabel(f'MAE (average over all times and outputs)')
            
            
            #st.pyplot()
            
            
            for name, value in multi_performance.items():
              print(f'{name:8s}: {value[1]:0.4f}, {value[2]:0.4f}, {value[3]:0.4f}, {value[4]:0.4f}')

            
            #st.subheader("Coeff determination")
            metric_name = 'coeff_determination'
            metric_index = multi_hybrid_model.metrics_names.index('coeff_determination')
            val_mae = [v[metric_index] for v in multi_val_performance.values()]
            test_mae = [v[metric_index] for v in multi_performance.values()]

            chart4 = get_determination(x, val_mae, width, test_mae, multi_performance)
            
            
            #st.pyplot(fig_r2)
            

            
            #st.header("Real data vs Neural Network Model Predictions [Normed]")
            
            
            chart5 = get_real_neural_normed(hybrid_pred, trues)
            #st.header('Real Data vs. Neural Network Model Predictions [KW]' )
            

            chart6 = get_real_neural_kw(hybrid_pred, trues, train_std, train_mean)
            
            hp = pd.DataFrame(hybrid_pred, columns = ['Predictions'])
            #json = hp.to_json(orient='records') 
            #columns = ['Predictions']

            json_records = hp.to_json(orient ='records') 
            data = [] 
            data = json.loads(json_records) 
            

            print(type(hp))
            return render(request, 'home.html', {'form': form, 
              'chart': chart,
              'chart1': chart1,
              'chart2': chart2,
              'chart3': chart3,
              'chart4': chart4,
              'chart5': chart5,
              'chart6': chart6,
              'd': data
              })
            #st.pyplot(fig)

            #st.header("Predictions")
            #st.write(hybrid_pred)

            
            ##############
            #return HttpResponseRedirect('/1/results/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = NameForm()
    #template_name = 'home.html'
    return render(request, 'home.html', {'form': form})


@login_required()
def predict(request): # let's say it's a contact form
    if request.method == 'POST': # If the form has been submitted...
        print(request.POST)
        # do your things with the posted data
    else: # form not posted : show the form.
        return render(request,'forecasting/home.html')



@login_required()
def dashboardView(request):
    return render(request,'forecasting/dashboard.html')


def registerView(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login_url')
    else:
        form = UserCreationForm()
    return render(request,'forecasting/register.html',{'form':form})


class DetailView(generic.DetailView):
    model = Question
    template_name = 'forecasting/detail.html'


class ResultsView(generic.DetailView):
    model = Question
    template_name = 'forecasting/results.html'

@login_required()
def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'forecasting/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
    return HttpResponseRedirect(reverse('forecasting:results', args=(question.id,)))

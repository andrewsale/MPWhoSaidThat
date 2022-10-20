import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sns
from time import time
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, log_loss
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

def fit_model(model_build_fn,
              X_train,
              X_val,
              y_train = None,
              y_val = None,
              n_repeats=1, 
              n_epochs=10, 
              early_stopping_patience = None,
              verbose=0,
              model_save_suffix='',
              **model_kwargs
             ):
    '''
    Takes a compiled model and data and fits it multiple times (with the same parameters).
    
    Args:
        model_build_fn:  function that builds and compiles TF model
        X_train:  input training data
        y_train:  training labels
        X_val:  input validation data
        y_val:  validation labels
        n_repeats=1: number of times to fit the model
        n_epochs=10, 
        early_stopping_patience = None
        **model_kwargs: specify any parameters to customize for the creation of the model 
        
    Returns:
        list of history classes
        
    File Outputs:
        model  --- saving the model from each run
    '''
    # Set up for Tensorboard - disabled in Kaggle
#     root_logdir = os.path.join(os.curdir, 'tb_logs')
#     def get_run_logdir():
#         run_id = strftime("run_%Y_%m_%d_%H_%M_%S")
#         return os.path.join(root_logdir, run_id)
#     run_logdir = get_run_logdir()
    
    # Prepare dict to log histories of the runs
    history_log = model_kwargs.copy()
    history_log['Parameters'] = str(model_kwargs)
    history_log['Parameter keys'] = model_kwargs.keys()
    history_log['Number of runs'] = n_repeats
    run_times = []
    
    for i in range(n_repeats):        
        # Build model
        model = model_build_fn(**model_kwargs)
        
        # Establish callbacks
        if not early_stopping_patience == None:
            early_stopping_monitor = EarlyStopping(patience=early_stopping_patience)
#             callbacks = [early_stopping_monitor, TensorBoard(run_logdir)]
            callbacks = [early_stopping_monitor]
        else:
#             callbacks = [TensorBoard(run_logdir)]
            callbacks = []
        model_filename = f'model{model_save_suffix}_run_{i}'
        model_save = ModelCheckpoint(model_filename, save_best_only=True, save_format='tf')
        callbacks.append(model_save)
    
        # Fit model  
        t0=time()
        history = model.fit(X_train, 
                            y_train,
                            epochs=n_epochs, 
                            validation_data=X_val, #(X_val, y_val), 
                            callbacks=callbacks, 
                            verbose = verbose)
        # Append the time for this run to the list
        run_times.append((time()-t0)/60)
        
        # Get vocab size
        history_log['Vocab size'] = model.get_layer(name='Vectorizer').vocabulary_size()
        
        # Make preditictions on training and validation sets
        model = tf.keras.models.load_model(model_filename)
        history_log[f'y_train_pred_run_{i}'] = model.predict(X_train)
        history_log[f'y_val_pred_run_{i}'] = model.predict(X_val)
        
        # Get loss and accuracy on val set when using the full speeches
        full_val_loss, full_val_accuracy = val_performance(model)
        history_log[f'y_full_val_loss_run_{i}'] = full_val_loss
        history_log[f'y_full_val_accuracy_run_{i}'] = full_val_accuracy
        
        # the following will extract the actual labels from a tf dataset when this is used
        if y_train == None:
            y_train1 = np.concatenate(list(X_train.map(lambda x,y:y).as_numpy_iterator()), axis=0)
        if y_val == None:
            y_val1 = np.concatenate(list(X_val.map(lambda x,y:y).as_numpy_iterator()), axis=0)
        history_log[f'y_train_actual'] = y_train1
        history_log[f'y_val_actual'] = y_val1
        # On the first run add the number of trainable parameters to the log dict
        if i==0:
            set_of_trainable_weights = model.trainable_weights
            trainable_count = int(sum([tf.keras.backend.count_params(p) for p in set_of_trainable_weights])) # counts trainable variables
            history_log['trainable parameters'] = trainable_count
            
        # Add the history of the current run to the log dict
        keys = history.history.keys()
        for key in keys:
            history_log[f'run_{i}_{key}'] = history.history[key]
        
        # Get val_loss trend over last 6 epochs
        slopes = []
        for epoch in range(n_epochs-6,n_epochs):
            dy = history.history['val_loss'][-7] - history.history['val_loss'][-1]
            dx = n_epochs - epoch
            slopes.append(dy/dx)
        history_log[f'Val_loss trend at end run {i}'] = np.mean(slopes)
        history_log['Number of epochs'] = len(history.history['val_loss'])
    
    # Get mean end trend
    history_log[f'Val_loss mean trend at end'] = np.mean([history_log[f'Val_loss trend at end run {i}'] for i in range(n_repeats)])
    
    # Add the run time data to the log dict
    history_log['Mean run time (mins)'] = sum(run_times)/len(run_times)
    
    # Collect stats of the runs in the log dict
    for key in keys:        
        final_key_values = [history_log[f'run_{i}_{key}'][-1] for i in range(n_repeats)]
        best_key_values = [max(history_log[f'run_{i}_{key}']) for i in range(n_repeats)]

        history_log[f'mean_{key}'] = sum(final_key_values) / n_repeats
        if 'loss' in key:
            history_log[f'best_final_{key}'] = min(final_key_values)
            history_log[f'best_anytime_{key}'] = min(best_key_values)            
        else:
            history_log[f'best_final_{key}'] = max(final_key_values)
            history_log[f'best_anytime_{key}'] = max(best_key_values)
        history_log[f'std_final_{key}'] = (sum(
            [(history_log[f'mean_{key}'] - x)**2 for x in final_key_values]) / n_repeats) ** 0.5  
    full_val_losses = [history_log[f'y_full_val_loss_run_{i}'] for i in range(n_repeats)]
    full_val_accuracies = [history_log[f'y_full_val_accuracy_run_{i}'] for i in range(n_repeats)]
    history_log['best_anytime_full_val_accuracy'] = max(full_val_accuracies)
    history_log['best_anytime_full_val_loss'] = min(full_val_losses)
    history_log['mean_full_val_loss'] = np.mean(full_val_losses)
    history_log['mean_full_val_accuracy'] = np.mean(full_val_accuracies)
        
    return history_log

def grid_search(model_build_fn,
                X_train,
                X_val,
                y_train=None,
                y_val=None,
                n_repeats=1, 
                n_epochs=10, 
                early_stopping_patience = None,
                verbose=0,
                **model_kwargs
               ):
    '''
    Runs fit_model for each parameter combination for those provided in model_kwargs.
    Provide each model_kwarg as a list of suitable objects
    '''
    param_combinations = []
    total_combs = 1
    number = {}
    for param in model_kwargs.keys():
        number[param] = len(model_kwargs[param])
        total_combs *= number[param]
    for i in range(total_combs):
        this_comb = {}
        cum_num=1
        for param in number.keys():
            n = int(i/cum_num)
            this_comb[param] = model_kwargs[param][n % number[param]]
            cum_num *= number[param]
        param_combinations.append(this_comb)
    history_logs = []
    for i, kwargs in enumerate(param_combinations):
        history_log = fit_model(model_build_fn,
                                X_train,
                                X_val,
                                y_train=y_train,
                                y_val=y_val,
                                n_repeats=n_repeats, 
                                n_epochs=n_epochs, 
                                early_stopping_patience = early_stopping_patience,
                                verbose=verbose,
                                model_save_suffix=f'_params_{i}',
                                **kwargs
                               )
        history_logs.append(history_log)
    return history_logs

def confusion_matrices(history_log, title=''):
    def prob_to_pred(x):
        if x < 0.5:
            return 0
        else:
            return 1
        
    def confmat_to_plot(confmat, axis, subtitle=''):
        sns.heatmap(confmat, annot=True, fmt='d', cbar=False, linewidths=.5, ax=axis)
        axis.set_xticklabels(['Predicted Tory', 'Predicted Labour'])
        axis.set_yticklabels(['Actual Tory', 'Actual Labour'])
        axis.set_title(subtitle)
        
    # Extract predictions from log into df
    train_preds = { f'y_train_pred_run_{i}' : [prob_to_pred(x) for x in history_log[f'y_train_pred_run_{i}']]  for i in range(history_log['Number of runs']) }
    val_preds = { f'y_val_pred_run_{i}' : [prob_to_pred(x) for x in history_log[f'y_val_pred_run_{i}']]  for i in range(history_log['Number of runs']) }
    
    # Print classification reports
    for i,run in enumerate(train_preds.keys()):
        print('-----------------------')
        print(f'Training set, run {i}:')
        print('-----------------------')  
        print(classification_report(history_log[f'y_train_actual'], train_preds[run]))  # print classification report
    
    for i,run in enumerate(val_preds.keys()): 
        print('-------------------------')
        print(f'Validation set, run {i}:')
        print('-------------------------')  
        print(classification_report(history_log[f'y_val_actual'], val_preds[run]))  # print classification report
        
    # Make plots for train + val set runs
    fig, ax = plt.subplots(nrows=2,ncols=history_log['Number of runs'], figsize = (4*history_log['Number of runs'], 8) )
    if history_log['Number of runs'] == 1:
        ax = ax.reshape((2,1))
    # Make plots for train runs
    for i,run in enumerate(train_preds.keys()):
        confmat_to_plot(confusion_matrix(history_log[f'y_train_actual'], train_preds[run]), axis=ax[0,i], subtitle=f'Training set, run {i}')
            
    # Make plots for val set runs
    for i,run in enumerate(val_preds.keys()):        
        confmat_to_plot(confusion_matrix(history_log[f'y_val_actual'], val_preds[run]), axis=ax[1,i], subtitle=f'Validation set, run {i}')
    plt.suptitle(title)
    plt.show()

def visualize_log(history_log_list):
    '''
    Visualizes the history of th grid search.
    
    Args:
        history_log_list : list of data from grid search
        input_format: use 'tf' for tf.data.Dataset input, 'pd' for pandas dataframe.
    '''
    # Establish the df and display parameters for reader
    print('-----------------------\nParameter lookup table:\n-----------------------')
    log_df = pd.DataFrame(history_log_list)
    log_df.index.name = 'Parameter Index'
    params_cols = set()
    for i in range(len(log_df)):
        params_cols = params_cols.union(set(log_df.loc[i,'Parameter keys']))
    other_cols = ['trainable parameters','mean_val_accuracy','best_anytime_val_accuracy', 'Mean run time (mins)',
                 'best_anytime_full_val_loss','best_anytime_full_val_accuracy',
                 'mean_full_val_loss','mean_full_val_accuracy', 'Val_loss mean trend at end',
                 'Number of runs', 'Number of epochs']
    print(log_df[list(params_cols)+other_cols])
    log_df  = log_df.reset_index()
    log_df.to_csv('training_log_df.csv')
    
    # Bar charts with best accuracies    
    fig = px.bar(data_frame=log_df, x=range(len(log_df)), y='mean_val_accuracy', hover_name='Parameter Index', title='Comparing mean validation accuracies')
    fig.show()
    fig = px.bar(data_frame=log_df, x=range(len(log_df)), y='best_anytime_val_accuracy', hover_name='Parameter Index', title='Comparing optimum validation accuracies')
    fig.show()
    
    # Loss plots
    def metric_df(metric='loss', run=0):
        padded_losses_dict = {}
        length = max([len(log_df.loc[i,f'run_{run}_{metric}']) for i in range(len(log_df))])
        for i in range(len(log_df)):
            this_loss = log_df.loc[i,f'run_{run}_{metric}']
            while len(this_loss) < length:
                this_loss.append(np.NaN)
            padded_losses_dict[log_df.loc[i,'Parameter Index']] = this_loss
        return pd.DataFrame( data=padded_losses_dict )
    
    for metric in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
        for run in range(log_df['Number of runs'].min()):
            fig = px.line(data_frame=metric_df(metric=metric, run=run),
                    title=f'Curves for {metric} of run {run} for each parameter set'
                   )
            fig.update_layout(legend_title_text='Parameter index')
            fig.show()
    
    # Show confusion matrices
    for history_log in history_log_list:
        confusion_matrices(history_log, 
                           title=f'Confusion matrix for {history_log["Parameters"]}'
                          )        

def val_performance(model):
    full_val_df = pd.read_csv('./Processed_speeches/full_val.csv', index_col=0)
    full_X_val = full_val_df['Speech']
    full_y_val = full_val_df['Label']
    probs = []
    preds = []
    for speech in tqdm(full_X_val):
        pred = speech_predict(speech, model, overlap=50)
        probs.append(pred)
        if pred < 0.5:
            preds.append(0)
        else:
            preds.append(1)

    loss = log_loss(full_y_val, probs)
    accuracy = accuracy_score(full_y_val, preds)
    return loss, accuracy

def load_data(batch_size=32):
    train_df = pd.read_csv('./Sampled_speeches/train.csv')
    val_df = pd.read_csv('./Sampled_speeches/val.csv')
    test_df = pd.read_csv('./Sampled_speeches/test.csv')

    train_ds = tf.data.Dataset.from_tensor_slices((train_df.Speech.values, train_df.Label.values)).cache().shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((val_df.Speech.values, val_df.Label.values)).cache().shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((test_df.Speech.values, test_df.Label.values)).cache().shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds

def speech_predict(speech, model, overlap = 50):
    split_speech = speech.split()
    length = len(split_speech)
    if length < 100:
        segments = [speech]
    else:
        segments = []
        i=0
        while i <= length - 100:
            segments.append(' '.join(split_speech[i:i+100]))
            i+=(100-overlap)
    preds = model.predict(segments, verbose=0)
    pred = np.sum(preds)/len(preds)
    return pred
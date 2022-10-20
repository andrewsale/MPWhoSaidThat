# MPWhoSaidThat
 A deep learning model that predicts whether a speech by a UK MP is by a Labour or Conservative politician


## About the app

In the app you can enter the text of a real speech given by a politician, or make up your own. It will then predict if the speech was given by someone from the left-leaning Labour Party, or the right-leaning Conservative Party.

<a href='https://rebrand.ly/whosaidthat', target=”_blank”>Click here to see the app in action.</a>

## The AI behind the app

A neural network was trained on samples from speeches scraped from <a href='https://www.ukpol.co.uk/'>www.ukpol.co.uk/</a> and <a href='http://www.britishpoliticalspeech.org/speech-archive.htm'>www.britishpoliticalspeech.org</a>.

The neural network is an RNN using LSTM modules. After vectorizing the text input and embedding this into $\mathbb{R}^n$, it is fed through a convolutional layer before being sent to the LSTM layers, and then onto some fully connected layers.

Adding the convolutional layer in the model noticable improved performance. Essentially it means the input to the LSTM layers is no longer a sequence of single words, but rather features built from short phrases in the speech.

## Short speeches

The neural network was trained on samples extracted from speeches. The samples are mostly of length 100 words, but some were deliberately made shorter to improve performance when the model is faced with limited input. It improved, but is still not where it should be. To perform well, the model needs input of a decent length (ideally more than 100 words).

When a speech of length more than 100 words is fed to the model, the speech is first broken up into 100 word chunks. The output probabilities are then averaged to get an overall probability, and hence a prediction.

## Files

* `main.py`: this is the script that runs the streamlit app.
* `helper_functions.py`: the name says it all. This contains functions used mostly when tuning the model.
* `model_tuning.ipynb`: notebook used when tuning. Hyperparameters can be set, the output visualized and saved.
* `scraping.py`: the script used to scrape the speeches from the two websites.
* `wrangling.ipynb`: the principal task performed by this notebook is to assign speakers (and hence parties) to each speech.
* `create_samples.ipynb`: notebook used to take random samples from the scraped speeches and create train, validation and test sets (this split was done on the speech level, before sampling, to avoid data leakage). 
* `test_yourself.py`: a second streamlit app that allows you to test your performance at recognizing speeches as Labour or Conservative.
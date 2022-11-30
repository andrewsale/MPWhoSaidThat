# MP - WhoSaidThat
 A deep learning model that predicts whether a speech by a UK MP was given by a Labour or Conservative politician


## About the app

In the app you can enter the text of a real speech given by a politician, or make up your own. It will then predict if the speech was given by someone from the left-leaning Labour Party, or the right-leaning Conservative Party.

<a href='https://andrewsale.github.io/MP-app.github.io/'>Click here to see the app in action.</a>

<a href="https://medium.com/@andrew.w.sale/deploying-text-classification-from-keras-to-tensorflow-js-92c614dca4ec">Click here to read the blog post.</a>

## The AI behind the app

A neural network was trained on samples from speeches scraped from <a href='https://www.ukpol.co.uk/'>www.ukpol.co.uk/</a> and <a href='http://www.britishpoliticalspeech.org/speech-archive.htm'>www.britishpoliticalspeech.org</a>.

The neural network has convolutional layers, followed by LSTM layers. After tokenizing the text input and embedding this into $\mathbb{R}^n$, it is fed through a convolutional block, with a skip connection, before being sent to the LSTM layers. Finally, it is sent through some fully connected layers.

Adding the convolutional layer in the model noticably improved performance. Essentially it means the input to the LSTM layers is no longer a sequence of single words, but rather features built from short phrases in the speech.

Numerous models were trialed. The final one (v5) showed good performance (88.5% accuracy on the validation set) and could be converted to a tensorflowjs mdoel for deployment.

## Training data

The neural network was trained on samples extracted from speeches. The samples are mostly of length 100 words, but some were deliberately made shorter to improve performance when the model is faced with limited input. Performance on short input improved, but is still not where we would like it to be. To perform well, the model needs input of a decent length.

When a speech of length more than 100 words is fed to the model, the speech is first broken up into 100 word chunks. The output probabilities are then averaged to get an overall probability, and hence a prediction.

## Files

* `create_samples.ipynb`: Notebook used to take random samples from the scraped speeches and create train, validation and test sets (this split was done on the speech level, before sampling, to avoid data leakage). 
* `model_to_js.py`: A short script to convert keras models to tensoflowjs.
* `model_tuning.ipynb`: Notebook used when tuning. Hyperparameters can be set, the output visualized and saved.
* `scraping.py`: The script used to scrape the speeches from the two websites.
* `tokenizer_dictionary.json`: The vocabulary file for the tokenizer.
* `wrangling.ipynb`: The principal task performed by this notebook is to assign speakers (and hence parties) to each speech.
* `Model-vX-Y-Z-W`: Folders containing keras models from various iterations of the model. Version numbering refers to the Kaggle notebook (X), its version number (Y), the parameter index (Z) and the run number (W).
* `Model_js`: Folder containing the tensoflowjs model.



## Kaggle links

The models were trained and the scripts run on the Kaggle server. I shared the dataset and notebooks there too.

* <a href='https://www.kaggle.com/datasets/andrewsale/uk-political-speeches'>Dataset</a>
* <a href='https://www.kaggle.com/code/andrewsale/speech-scraping'>Scraping notebook</a>
* <a href='https://www.kaggle.com/code/andrewsale/speeches-data-wrangling'>Wrangling notebook</a>
* <a href='https://www.kaggle.com/code/andrewsale/speeches-sampling'>Sampling notebook</a>
* <a href='https://www.kaggle.com/code/andrewsale/speeches-classification-model-trials-v5/notebook'>Model tuning notebook</a>

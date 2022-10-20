import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
from bs4 import BeautifulSoup
import re
import string
from time import time

def get_ukpol_speech(url, lastname_first_letter=None):
    # Get the html and soup it
    try:
        r = requests.get(url)
    except:
        print(f'Failed to parse: {url}')
        return
    # Obtain the division of the page with the speech
    soup = BeautifulSoup(r.text,"lxml")
    div = soup.find(name='div', attrs={'class': 'entry-content'})
    if div == None:
        print(f'No soup found. Check url: {url}')
        return
    speech_div = div.find_all(name='p')
    
#     print(speech_div)
    
    # find the desription of the speech
    desc_index=-1
    for j in range(len(speech_div)):
        if speech_div[j].em:
            desc_index = j
            break
    if desc_index==-1:
        print(f'Description not found for {url}.')
        return
    speech_description = str(speech_div[desc_index].string)
    
    # extract the speech as one long string.
    speech = ''   
    faulty_encoding_samples = {'â\x80\x9c':'"', 'â\x80\x9d':'"', 'â\x80\x94':' ', 'â\x80\x99':"'", '\xa0':'', '\x80\x94':'', '\xe2\x80\x99':"'", '\xe2\x80\x93':'-'}
    for i in range(desc_index+1,len(speech_div)): 
        speech += ' '
        next_sentence = str(speech_div[i].string)
        for fault in faulty_encoding_samples.keys():
            next_sentence = next_sentence.replace(fault, faulty_encoding_samples[fault])
        if re.match(r'\s+\[insert_php\]', next_sentence):
            next_sentence = ''
        speech += next_sentence
    
    if re.search('\Wx', speech):
        print(f'Possible encoding error found in {url}')
        
    # find the tags
    tag_soup = soup.find(name='span', attrs={'class':'tag-links'})
#     tags = []
    candidate_year=None
    candidate_speaker=None
    try:
        for a in tag_soup.find_all('a'):
            text = a.get_text()
    #         tags.append(text)
            if text.isdigit():
                candidate_year = int(text)
            elif candidate_speaker != None:
                if not lastname_first_letter in [x[0].lower() for x in candidate_speaker.split()]:
                    candidate_speaker = text
            elif candidate_speaker == None:
                if not text in ['Nelson Mandela','Sinn Fein','Taxation','Corbyn Suspended','Speeches', 'Comments', 'Constituency', 'Department for Education', 'Foreign and Commonwealth Office', 'Press Release', 'Foreign Office', 'HM Queen Elizabeth II', 'Maiden Speech', 'President of Ukraine', 'Press Release']:
                    candidate_speaker = text
    except AttributeError:
        print(f'No tags found for {url}')
    return (candidate_year, candidate_speaker, speech_description, speech.strip())

def fetch_ukpol_speeches(start_index = 0, letters_to_do = 26):
    # Find the URLs and grab the speeches
    descriptions = []
    speeches = []
    years = []
    speakers = []
    alphabet = list(string.ascii_lowercase)
    for a in alphabet[start_index: start_index + letters_to_do]:
        t0=time()
        if a == 'c':
            url_a = 'https://www.ukpol.co.uk/speeches/c/'
        else:
            url_a = f'https://www.ukpol.co.uk/speeches/speeches-{a}/'
        try:
            r = requests.get(url_a)
        except:
            print(f'Failed to parse: {url_a}')
            break
        soup = BeautifulSoup(r.text,"lxml")
        div = soup.find(name='div', attrs={'class': 'entry-content'})
        div = div.find_all(name='a', href=True)
        print(f'There are {len(div)} speeches for letter {a}.')
        for i in range(len(div)):
            new_url = div[i]['href']
            x = get_ukpol_speech(new_url, lastname_first_letter=a)
            if not x == None:
                year, speaker, desc, speech = x 
            descriptions.append(desc)
            speeches.append(speech)
            years.append(year)
            speakers.append(speaker)
        print(f'Done all from letter {a} in {time()-t0:.2f} seconds.')
    return pd.DataFrame({"Speaker":speakers, "Year":years, "Description":descriptions,"Speech":speeches})

def get_bps_speech(url):
    # Get the html and soup it
    try:
        r = requests.get(url)
    except:
        print(f'Failed to parse: {url}')
        return
    
    soup = BeautifulSoup(r.text,"lxml")
    # Extract the speech
    speech_soup = soup.find(name='div',attrs={'class':'speech-content'})
    speech = speech_soup.get_text()
    replace_strings = ['\n\xa0\n', '\n', '\xa0']
    for string in replace_strings:
        speech = speech.replace(string, ' ')
    # Now find the speaker
    speaker_soup = soup.find(name='p', attrs={'class':'speech-speaker'})
    speaker = speaker_soup.get_text()
    
    return speech, speaker   

def fetch_bps_speeches():
    index_url = 'http://www.britishpoliticalspeech.org/speech-archive.htm'
    r = requests.get(index_url)
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find_all(name='tbody')[1]
    dates = []
    parties = []
    speakers = []
    speeches = []
    for row in table.find_all(name='tr'):
        current = row.find_all('td')
        date = current[0].get_text()
        party = current[1].get_text()
        speaker = current[2].get_text()
        url = 'http://www.britishpoliticalspeech.org/' + current[3].a.get('href')
        speech, speaker_alt = get_bps_speech(url)
        if len(speaker) == 0:
            speaker = speaker_alt
        dates.append(date)
        parties.append(party)
        speakers.append(speaker)
        speeches.append(speech)
    return pd.DataFrame({"Speaker":speakers, "Date":dates, "Party":parties,"Speech":speeches})


speech_df = fetch_ukpol_speeches(start_index = 0, letters_to_do = 26)
speech_df.to_csv('Raw_speeches/speeches_ukpol.csv')

speech_df1 = fetch_bps_speeches()
speech_df1.to_csv('Raw_speeches/speeches_bps.csv')
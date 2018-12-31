import pandas as pd
import numpy as np
import re
import random
import sys

from keras import layers
from keras.models import Sequential
from keras import optimizers


# -----------------------------------------------------------------------
# Get Lyrics From a Category
# -----------------------------------------------------------------------


def clean_str (s):
    s = s.lower()
    s = re.sub('[\s]', ' ', s)
    return s


def create_str(df):
    """
    Create one big string with all the lyrics together.
    
    :param df: Dataframe of lyrics
    :type  df: pandas.core.frame.DataFrame
    :returns: string
    :rtype:   str
    """
    
    lyrics = ''
    for song in df['cleaned lyrics']:
        lyrics += song
    return lyrics


def get_lyrics_of_category(category, description):
    """
    Get all the lyrics from a certain category meeting a description.

    Example: 
        category    Description
        year        2010
        artist      Drake
        genre       Pop
    
    :param category: A column name from data
    :type  category: str
    :param description: A subset of category
    :type  description: str
    :param data: Dataframe of songs
    :type  data: pandas.core.frame.DataFrame
    :returns: A string with all the lyrics together.
    :rtype:   str
    """
    
    lyric_data = data[data[category] == description]
    lyric_data['cleaned lyrics'] = lyric_data['lyrics'].apply(
        lambda x: clean_str(x)
    )
    lyrics = create_str(lyric_data)
    return (lyrics)


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------


def vectorizing_seq (text, maxlen, step):    
    """
    :param maxlen: the length of a sequence to extract as train
    :type  maxlen: int
    :param step: sample a new sequence every n steps
    :type  step: int
    :returns: (Numpy boolean array of shape 
                    (Number of sequences, maxlen, number of distinct character),
               Numpy boolean array of shape 
                    (Number of sequences, number of distinct character),
               dictionary mapping a character to its integer placeholder)
    :rtype:   (numpy.ndarray, 
               numpy.ndarray, 
               dict)     
    """
    
    sentences = [] # hold extracted sequences
    next_chars = [] # hold next characters for each corresponding sentence

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    print('Number of sequences:', len(sentences))

    chars = sorted(list(set(text)))
    print('Unique characters:', len(chars))
    char_indices = dict((char, chars.index(char)) for char in chars)
    print('Vectorization...')

    # one hot encoding the characters into binary arrays
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
        
    return x, y, char_indices


def create_model(x, y, maxlen, epochs, chars):
    """
    Creates and trains a model.
    :param x: Numpy boolean array of shape 
                    (Number of sequences, maxlen, number of distinct character)
    :type  x: numpy.ndarray
    :param y: Numpy boolean array of shape 
                    (Number of sequences, number of distinct character)
    :type  y: numpy.ndarray
    :param maxlen: the length of a sequence to extract as train
    :type  maxlen: int
    :param epochs: number of training iterations
    :type  epochs: int
    :param chars: list of unique characters
    :type  chars: list
    :returns: trained keras model
    :rtype:   keras.engine.sequential.Sequential
    """

    model = Sequential()
    model.add(layers.GRU(
        32,
        return_sequences=True,
        input_shape=(maxlen, len(chars)))
    )
    model.add(layers.GRU(
        64,
        input_shape=(maxlen, len(chars)))
    )
    model.add(layers.Dense(
        len(chars), 
        activation='softmax')
    )

    print(model.summary())

    optimizer = optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.fit(x, y, batch_size=128, epochs=epochs)

    return (model)


def train_model_from_lyrics(lyrics, maxlen=60, step=20, epochs=10):
    """
    Given lyrics, train the model.
    
    :param lyrics: A string with all the lyrics together.
    :type  lyrics: str
    :param maxlen: the length of a sequence to extract as train
    :type  maxlen: int
    :param step: sample a new sequence every n steps
    :type  step: int
    :param epochs: number of training iterations
    :type  epochs: int
    :returns: (trained keras model,
               dictionary mapping characters to digit representations)
    :rtype:   (keras.engine.sequential.Sequential,
               dict)
    """
    
    x, y, char_indices = vectorizing_seq(lyrics, maxlen, step)
    chars = list (char_indices.keys())
    model = create_model(x, y, maxlen, epochs, chars)
    
    return model, char_indices


# -----------------------------------------------------------------------
# Text Generation
# -----------------------------------------------------------------------


def sample(preds, temperature=1.0):
    """
    Compute new probability distribution based on the temperature
    Higher temperature creates more randomness.
    
    :param preds: numpy array of shape (unique chars,), and elements sum to 1
    :type  preds: numpy.ndarray
    :param temperature: characterizes the entropy of probability distribution
    :type  temperature: float
    :returns: a number 0 to the length of preds - 1
    :rtype:   int
    """
    
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def text_generate(model, text, char_indices, maxlen=60, temperature=1.0, textlen=400):
    """
    Generate text based on a model.
    
    :param model: trained keras model
    :type  model: keras.engine.sequential.Sequential
    :param text: lyrics
    :type  text: str
    :param char_indices: dictionary mapping a character to its integer placeholder
    :type  char_indices: dict
    :param maxlen: maximum length of the sequences
    :type  maxlen: int
    :param textlen: Number of characters of generated sequence
    :type  textlen: int
    """

    start_index = random.randint(0, len(text) - maxlen - 1) 
    generated_text = text[start_index: start_index + maxlen] 
    print('--- Generating with seed: "' + generated_text + '"')
    
    chars = list (char_indices.keys())
    
    print('------ temperature:', temperature)
    sys.stdout.write(generated_text)
    for i in range(textlen):
        sampled = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(generated_text):
            sampled[0, t, char_indices[char]] = 1
        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = chars[next_index]
        generated_text += next_char
        generated_text = generated_text[1:]
        sys.stdout.write(next_char)


# raw data
data = pd.read_csv('data/lyrics.csv')
data.drop(['index'], axis=1, inplace=True)
data.dropna(inplace=True)

# names of all the artists
artists = list (data['artist'].unique())

# all the years
years = list (data['year'].unique())

# all the genres
genres = list (data['genre'].unique())

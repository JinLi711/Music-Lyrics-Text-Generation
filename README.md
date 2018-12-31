# Music Lyrics Text Generation

This repository generates lyrics based on a singer's writing style. 
It can also be used to generate lyrics based on a year or based on a genre.

## Visualization

Check out some of the graphs in visualization.ipynb

## Preprocess

I lowercased all the text, then used one hot encoding to convert characters to arrays.

## Model

I used a two layer GRU model, and then stacked it with a dense layer. The activation function for the dense layer is softmax, the loss function is categorical crossentropy, and the optimizer is RMS.

### Training

This text generation is character based, where the input is a sequence of characters and the target is the character right after the sequence.

## Text Generation

Start with a small sequence of text. Use the trained model to create a probability distributions of possible characters. Redistribute that probability distribution to either increase or decrease entropy. Randomly pick one letter from that probability distribution. 

### Sample text generation

"CHARACTERS" is the seed.

#### Bob Dylan

"lot of them seemed to be lookin' my way brownsville girl wit"h the wordleed to me the brefamy back anch me when the deass on that he can a walked me can tell be the core see the cornever and camb me, go and he dres well, he said efave no have and get so was was was when you can to waty is the there's no but i come bury the wind well,

#### Beatles

"o yeah well, that much is alright well, that much is alright"ea trieve in a litt. you wormwere's numer jahd. now i'ver not bett you well you shound nur. nah, oh, you make mire, band good ne) ounn and i'm soong that is can durr of thisttly get that chat i need sinde is here jold a cool the is a and i do and i need a crould mus johnn tell me have and she's be and i you cear i hear that you what'gound hard me ah ay la,. don't you hear true shoe'm a know,

## Problems

Since I only trained each model for about ten minutes, the text that is generated does not make a lot of sense.

## Solutions

Increase the training epochs or decrease the steps. This will create more accurate models, but will take longer.

## Things To Work On

1. Next time, create a word based text generation instead of character based. Also, use embedding instead of one hot encoding.
2. Somehow find a way to make comparisons between different singers.


# Character Handwriting Predictor
Basic program that allows you to write a number (0-9), lowercase (a-z), or uppercase (A-Z) letter, and attempts to predict what was written.
Implemented mainly using the PyTorch libraries.

## Features
- Uses a convolutional neural network to train on the EMNIST dataset.
- Provides a simple interface that lets you write, clear the board, and predict the character.

## Dataset
The model was trained using the [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset), an extension of the MNIST dataset.
I downloaded the dataset in Matlab format, and put the 'emnist-byclass.mat' file in the same directory as the two scripts.

## Future improvements
- Can improve the model by:
  - Further tuning hyperparameters
  - Using a more complex model
  - Training for more epochs
  - Data augmentation
 - Extend to a wider range of characters
 - Allow for 'erasing' when writing.
 - Extend to Japanese characters
  
## Motivations
This was created as a prerequisite to creating a similar model, but for Japanese characters (hiragana, katakana, kanji) to aid in my study.

## Citations
Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373

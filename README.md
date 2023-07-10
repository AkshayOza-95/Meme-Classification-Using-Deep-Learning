# Meme Classification Using and Fine Tuning VGG-16 Model

## Background

Toady, memes have become a new language for millenials on social media. With billions of memes
out there- political, funny, informative, fan theory, etc. people have been sharing them on social
media majorly for entertainment. At times these memes have a purpose beyond entertainment.
They are shared to spread information and sentiments regarding political, economic and social
issues. This also leads to spread of disinformation. With today’s advancement in technology,data
science and AI, it is possible to identify if these memes are facts or just mere wrong information.
However, before identifying that it is necessary to build a model that can classify memes and nonmeme
images. In this project I used pre-trained VGG-16 neural net models and fine tuned them
to increase their accuracy in classifying memes and non-meme images.

## Data

The data used in this project was scraped and downloaded from various sources. A total of 16000
images (8000 images for each class) have been used in training the models. The image sources are
Bing Image Search and Kaggle Dataset. For Bing Image Search , I used a bing images python script
with different arguments.The images were then renamed as ‘meme_#.jpg’ and ‘non_meme_#.jpg’
(# is the number) using a python script. To make sure there were no duplicates or false memes /
non-memes I manually checked for duplicates and deleted them.

**Note: The data is not provided since it was proprietary for this project** 

## Required Libraries and GPU

I used Google Colab for this project. I saved the files on my Google Drive and them mounted it to the Colab notebook. I used the free GPU provided by Colab for training
and fin-tuning the deep learning VGG-16 model. I used the keras API from Tensorflow. 

## Conclusion

- The final model after hyperparameter tuning had 15 epochs, a batch size of 100, dropout rate of
0.2, learning rate of 0.001 and Adamax optimizer.
- VGG-16 model when fine tuned gives great accuracy and minimal loss for training and
validation dataset as well as test data. The final model provides a Precision of 97% and a
recall of 96.75% when used to classify test data.

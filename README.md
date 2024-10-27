# Scam-Email-Detection-NewHacks-2024 (Phishing Nemo)

## Recurrent Neural Network
After implementing previous models to lackluster results and heavy overfitting, a Recurrent Neural Network was chosen, following several guides in order to change the goal of the project, and detect spam emails. This was done through embedding of individual characters, and a Gated Recurrent Unit, as well as through small improvements such as the BucketIterator, and an improved Adam optimizer and Binary Cross Entropy with Logits loss function. An average test accuracy of 91% was observed after only 5 epochs.

## Chrome Extension & Front End
Front end website hosted at **findingnemo.co** *https://github.com/gracelhao/gracelhao.github.io* developed with **React**. A chrome extension is also available with a download on the website, and users can click to scan their email.

## Back End & RNN Implementation
Hosted with Flask to direct user emails to a preloaded version of the RNN, trained on similar data.


## Neural Network & Other Attempts
### *phishingDetection.ipynb*
> Created a custom Dataset type to feed inputs through a simple Neural Network implementing a Stochastic Gradient Descent and a Cross Entropy Loss function.
Initial attemps were built using **PyTorch** in a **Jupyter** Notebook, including a simple Neural Network to identify FashionMNIST photos, attempts to investigate the NaïveBayes algorithm, and an early version of a Neural Network to detect phishing emails.
### *Neural Network Testing & NaïveBayes_ML_techniques*
> Contain initial Notebooks and FashionMNIST datasource as well as trained *.pkl* files
Dataset was pulled from *https://github.com/diegoocampoh/MachineLearningPhishing* taking over 2000 emails from the Enron database and over 2000 reported phishing emails.
Several data elements were extracted from each email to create *features-enron-and-phishing.csv* as well as passed to the function upon new calls.

### *features-enron-and-phishing.csv*
> Contains over 4500 entries of several data values from individual emails, as featured in *https://github.com/diegoocampoh/MachineLearningPhishing*
Including,
- @ in URLs. Detects the presence of the ``@" character in URLs. T/F value.
- Number of Attachments. Detects the number of attachments present in the email. Numerical value
- Css in header. Detects the number of Css links in the emails’ body, under the head tag in the html message. Numerical value
- External Resources. Detects the number of external resources linked in the body of the email. Numerical value
- Flash content. Detects the presence of flash content in the body of the email. T/F value.
- HTML content. Detects the presence of HTML content. T/F value.
- Html Form. Detects the presence of HTML Forms. T/F value.
- Html iFrame. Detects the presence of HTML iFrames. T/F value.
- IPs in URLs. Detects the presence of IPs in URLS, instead of human readable domain names. T/F value.
- Javascript blocks. Detects the number of Javascript blocks inside the email’s body.
- URLs. Detects the number of URLS in the email.

### *HTML_extraction.ipynb*
> Implements several functions to scrape an email as a string of html into the paramaters above using **Python** libraries such as **RegEx**, **BeautifulSoup4**, and **tldextract**.


# References
- https://github.com/fennybz/Detecting-Phishing-Attack-using-ML-DL-Models/tree/main
- https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c1f27856df9df9a09a259711bf00161013e298f4
- https://github.com/diegoocampoh/MachineLearningPhishing/tree/master

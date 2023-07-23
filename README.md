# CLiQS Python module

CLiQS Python module provides implementation of multilingual crisis social media summarization model.

Please, if you use CLiQS for your research consider citing:

>Fedor Vitiugin, Carlos Castillo: Cross-Lingual Query-Based Summarization of Crisis-Related Social Media: An Abstractive Approach Using Transformers. In ACM Hypertext 2022. ACM Press. https://doi.org/10.1145/3511095.3531279

## Installation

1. Install the module via pip:

```console
pip install cliqs
```

2. Download LASER and CLiQS models:

```console
python3 -m laserembeddings download-models
python3 -m cliqs download-models
```

3. Before running the script, please check installation of [SpaCy models](https://spacy.io/models) for language that you plan to use.


## Use

Example of use:

```console
import pandas as pd
from cliqs import CliqSum

sum = CliqSum()

tweets = pd.read_csv('resources/example.csv')
summary = sum(tweets, Damage, fr)

print(summary)

>> The output summary ...
```

- example.csv —- data file with three columns: id, text, en_text (translation of texts to English).
- Damage -- information category. Current version supports 6 categories: Casualties, Damage, Danger, Sensor, Service aand Weather.
- fr -- language of texts in file.

## Resources

Code for training custom models — [CLiQS-CM GitHub repository](https://github.com/vitiugin/CLiQS-CM)

Dataset for text classification — [tweets dataset](https://data.d4science.org/ctlg/ResourceCatalogue/cross-lingual_dataset_of_crisis-related_social_media)

Dataset for summary evaluation — [summaries dataset](https://data.d4science.org/ctlg/ResourceCatalogue/dataset_for_evaluating_abstractive_summaries_of_crisis-related_social_media)
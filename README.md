# CLiQS-CM Python module

CLiQS-CM Python module provides implementation of multilingual crisis social media summarization model.

Please, if you use CLiQS-CM for your research consider citing:

>Fedor Vitiugin, Carlos Castillo: Cross-Lingual Query-Based Summarization of Crisis-Related Social Media: An Abstractive Approach Using Transformers. In ACM Hypertext 2022. ACM Press. https://doi.org/10.1145/3511095.3531279

## Installation

1. Install necessary dependencies with use of pip:

```console
pip -r requirements.txt
```

2. Download LASER models:

```console
python3 -m laserembeddings download-models
```

3. [Download models](https://zenodo.org/record/7754714) and extract files in `resources` folder. Before run the module `resources` should contains `category_model` and `disaster_detect` folders with models.

4. Before running the script, please check installation of [SpaCy models](https://spacy.io/models) for language that you plan to use. You can manually add or change predifined model for target language by editing function `get_spacy_model` in [features_extraction.py](features_extraction.py).


## Use

Example of use:

```console
python3 main.py example.csv Damage fr
```

- example.csv —- data file with three columns: id, text, en_text (translation of texts to English).
- Damage -- information category. Current version supports 6 categories: Casualties, Damage, Danger, Sensor, Service aand Weather.
- fr -- language of texts in file.

## Resources

Code for training custom models — [CLiQS-CM GitHub repository](https://github.com/vitiugin/CLiQS-CM)

Dataset for text classification — [tweets dataset](https://data.d4science.org/ctlg/ResourceCatalogue/cross-lingual_dataset_of_crisis-related_social_media)

Dataset for summary evaluation — [summaries dataset](https://data.d4science.org/ctlg/ResourceCatalogue/dataset_for_evaluating_abstractive_summaries_of_crisis-related_social_media)
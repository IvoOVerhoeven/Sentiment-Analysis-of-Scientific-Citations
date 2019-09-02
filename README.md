# Sentiment-Analysis-of-Scientific-Citations
A replication study for an academic internship at University College Roosevelt.

The entire report has been included, describing the approaches taken while replicating. The largest difference between this replication and previous attempts is the emulation of WEKA's DictionaryBuilder class. Furthermore, the Naive Bayes model proved to be much better at predicting citation sentiment that first reported by Athar. The most important results have been summarized in a table below. Further extension with additional features did not result in greater predictive power.


| Data Set  | NB - Original | NB - Replication | SVM - Original | SVM - Replication |
| --- | --- | --- | --- | --- | 
| 1-3 gram  | 0.474/0.764 | 0.579/0.819 | 0.597/0.862 | 0.596/0.886 |
| FS1  | 0.469/0.755  | 0.664/0.863 | 0.760/0.897 | 0.721/0.909 |
| FS2  | 0.471/0.755  | 0.664/0.862 | 0.764/0.898 | 0.724/0.910 |

Numbers reported are the macro/micro averaged F1 scores.


Set working directory (R: setwd) to folder in which this folder is extracted.

For Athar replications, set readtext directories to corpora folder, and import Citation Sentiment Corpus v2

For Jha replication, idem, and import Citation Jha et al.

For Jha lexicon approaches, set readtext and read csv directories to wordlists folder, and import necessary files

Make certain to load (R: source) the Classification_Functions.R file first!

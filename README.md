# MULTIOPED:A Corpus of Multi-Perspective News Editorials

This repositary contains the code and data of ["MultiOpEe:A Corpus of Multi-Perspective News Editorials."](https://cogcomp.seas.upenn.edu/papers/LCUR21.pdf) in NAACL'21.


## The MultiOpEd Dataset

Our dataset follows the below structure

<img src="https://github.com/CogComp/MultiOpEd/blob/main/dataset%20structure.png" width=60% height=60%>

### Dataset statistics

The below two graphs show the statistics of our dataset and distribution of topics that our dataset covers.

<img src="https://github.com/CogComp/MultiOpEd/blob/main/dataset%20statistics.png" width=40% height=40%> <img src="https://github.com/CogComp/MultiOpEd/blob/main/topic%20distributions.png" width=40% height=40%>

<h2>Reproducing the results</h2>

To reproduce the result, download the stance and relevance classifier from this [google drive](https://drive.google.com/drive/folders/1tHmPTa6Ji0r8--j2ZIMjEMR3gg_JlSR8?usp=sharing), modify their path in eval.sh, and simply run sh eval.sh

This should reproduce exactly the same result as we show in the paper.

## Trained models

Our best trained model is also available in this [google drive](https://drive.google.com/drive/folders/1tHmPTa6Ji0r8--j2ZIMjEMR3gg_JlSR8?usp=sharing). It is a multi-task BART-based model that uses both relevance and stance classification tasks as auxiliary signals.


<h2>Train</h2>

Refer to train.py and train_both_auxiliary.py for training from scratch. More instructions of usage will be added soon.


## Citation

```
@inproceedings{LCUR21,
    author = {Siyi Liu and Sihao Chen and Xander Uyttendaele and Dan Roth},
    title = {{MultiOpEd: A Corpus of Multi-Perspective News Editorials}},
    booktitle = {Proc. of the Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
    year = {2021}
}
```

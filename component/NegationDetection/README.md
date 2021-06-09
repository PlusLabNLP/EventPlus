# Negation Cue Detection and Scope Resolution

The training, evaluation and inference code for NegBERT.

For the cue detection, the label for each word follows the annotation schema:
* 0: Affix
* 1: Normal cue
* 2: Part of a multiword cue
* 3: Not a cue

## Performance

Negation cue detection, evaluating on SFU review dataset:

```
Validation loss: 0.14660959019822234
Validation Accuracy: 0.9972840671132076
Validation Accuracy for Positive Cues: 0.9394110275689225
       1     2        3
1  733.0   1.0     35.0
2    1.0  30.0     11.0
3   75.0  14.0  48267.0
              precision    recall  f1-score   support

           1       0.91      0.95      0.93       769
           2       0.67      0.71      0.69        42
           3       1.00      1.00      1.00     48356

    accuracy                           1.00     49167
   macro avg       0.86      0.89      0.87     49167
weighted avg       1.00      1.00      1.00     49167

F1-Score: 0.9972513069839883
Precision: 0.8955399061032864
Recall: 0.9431396786155748
F1 Score: 0.9187236604455147
F1-Score Cue_No Cue: 0.9972891007811321
```

Negative scope resolution, evaluating on SFU review dataset:

```
Validation loss: 0.21461165494672807
Validation Accuracy: 0.9522214842258335
Validation Accuracy Scope Level: 0.7831683168316831
Precision: 1
Recall: 0.7838509316770186
F1 Score: 0.8788300835654596
              precision    recall  f1-score   support

           0       0.97      0.97      0.97     16358
           1       0.90      0.92      0.91      5279

    accuracy                           0.95     21637
   macro avg       0.94      0.94      0.94     21637
weighted avg       0.96      0.95      0.95     2163
```

## Training

To train the negation cue detection model, set `SUBTASK = 'cue_detection'`. For negation scope resolution model, set `SUBTASK = 'scope_resolution'`. Then

```
python train.py
```

## Acknowledgement

```
@article{Khandelwal2020NegBERTAT,
  title={NegBERT: A Transfer Learning Approach for Negation Detection and Scope Resolution},
  author={Aditya Khandelwal and Suraj Sawant},
  journal={ArXiv},
  year={2020},
  volume={abs/1911.04211}
}
```

Adapted from the codebase: https://github.com/adityak6798/Transformers-For-Negation-and-Speculation

The SFU review dataset can be downloaded from [this link](https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html)
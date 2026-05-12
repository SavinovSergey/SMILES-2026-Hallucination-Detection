# Solution

## Reproducibility

Run the solution from the repository root:

```bash
pip install -r requirements.txt
python3 solution.py
```

The run produces:

```text
predictions.csv
results.json
```

The implementation is deterministic at the level used in my experiments: fixed split seed, fixed model/probe seeds, fixed 
PCA/scaler fitting on the training part of each fold, and fixed threshold-selection logic. A CUDA GPU is recommended 
because hidden-state extraction is the slowest part. I ran the experiments in Kaggle on a T4 GPU.

Important files modified:

```text
aggregation.py
probe.py
splitting.py
```

## Final configuration

I selected the following configuration as the best accuracy-oriented 5-fold run:

```yaml
target_metric: accuracy
split_strategy: stratified_5fold_with_inner_val
n_folds: 5
seed: 42

aggregation:
  raw_layers: [-4, -8, -12, -16]
  trajectory_layers: [-8, -10, -12, -14, -16]
  token_window: 32
  token_offset: -2

preprocessing:
  scaler: StandardScaler
  dim_reduction: PCA
  pca_n_components: 128
  pca_fit: train_only

probe:
  type: log_reg_cv
  max_iter: 2000
  cv: 5
  regularization: L2

threshold:
  selected_by: validation_accuracy
```

Final averaged CV result:

```text
majority_baseline_accuracy = 0.7010
avg_test_accuracy          = 0.7416
avg_test_f1                = 0.8269
avg_test_auroc             = 0.7155
avg_train_accuracy         = 0.7882
avg_train_f1               = 0.8571
avg_train_auroc            = 0.8713
pca_dim                    = 128
```

I therefore do not claim that this is the best AUROC model. I selected it because it is the best configuration according 
to the target metric I optimized: accuracy.

## What the final approach does

### 1. Hidden-state aggregation

In the final extractor, I combine two types of information.

First, I use a raw hidden-state vector from selected intermediate layers:

```python
RAW_LAYERS = [-4, -8, -12, -16]
TOKEN_OFFSET = -2
```

Here, `TOKEN_OFFSET = -2` means that the raw hidden-state slice is read at `last_non_padding_index + TOKEN_OFFSET` 
(clamped to valid positions), i.e. one token *before* the last real token instead of the last real token itself.

The likely reason is practical rather than theoretical: the last real token is often an EOS/closing token or carries 
little semantic information, while the previous token is closer to the final content-bearing part of the answer. 
This change increased threshold-level classification accuracy even though AUROC decreased.

Second, I add a compact trajectory/geometric feature block over the last 32 real tokens:

```python
TRAJECTORY_LAYERS = [-8, -10, -12, -14, -16]
TOKEN_WINDOW = 32
```

These features summarize the behavior of hidden states across tokens and layers. My final compact set includes statistics 
such as token-vector norms, cosine similarity between layers, L2 drift between layers, last-token vs. window-mean 
differences, token-level variance, drift across the answer tail, and a small spectral summary of covariance structure.

My important conclusion was that the useful signal was not only in a single final hidden vector. Some signal appeared in the
geometry of how representations changed across tokens and intermediate layers. However, adding too many spectral features 
made the model worse, so I keep the compact trajectory block rather than the expanded INSIDE/EigenScore-like version in the 
final solution.

### 2. PCA and low-capacity probe

I therefore use this final probe pipeline:

```text
StandardScaler -> PCA(128) -> LogReg
```

### 3. Logistic regression instead of an MLP

Features are high-dimensional and labels are scarce, so a multi-layer probe overfits easily. PCA plus L2 logistic regression 
(LogisticRegressionCV) keeps capacity low, matches linear probing on top of the LM’s representations, and selects 
regularization automatically. The decision threshold is tuned for validation accuracy, aligned with the competition metric. 
Overall: stability and simplicity over extra nonlinearity in the probe.


## Experiments and conclusions

### Stage 1: baseline hidden states + MLP

I started with the final-layer last-token hidden state and an MLP. This gave me a misleading result: some single-split 
scores looked good, but the model overfit heavily.

My original high-capacity MLP could almost perfectly separate the training data. In one early run, train AUROC was close to 
1.0. I did not treat this as reliable because the dataset is small and the majority baseline is already around 70% accuracy.

My conclusion: raw final-layer last-token features contain signal, but a large probe can memorize the train split. 
Single-split results were not trustworthy enough.

### Stage 2: regularization and 5-fold evaluation

I then added a regularized MLP with AdamW and weight decay. One single split reached high accuracy, but the 5-fold result 
dropped close to the majority baseline:

```text
regularized_mlp single split: accuracy ≈ 0.7692
regularized_mlp 5-fold:      accuracy ≈ 0.7054
```

This was an important warning. I treated the single split as too optimistic. After that, I used stratified 5-fold evaluation
with an inner validation split as my main decision signal.

My conclusion: the final solution must be selected by 5-fold accuracy, not by one lucky split.

### Stage 3: pooling experiments

I tried to improve aggregation by using more token information: last pooling, mean pooling, and max pooling. This did not 
help. Increasing the representation from one final token to large pooled vectors mostly added noise and made overfitting 
worse.

My conclusion: naive pooling across all tokens is not useful here. The answer-level signal must be added in a more compact 
and structured way.


### Stage 4: PCA compression

Because the multi-layer raw vector was high-dimensional, I tried PCA.

My main observations were:

```text
PCA-32:  underfit; train/test gap decreased but useful signal was removed
PCA-64:  better than PCA-32 but still weaker
PCA-128: best balance among PCA variants
PCA-256: did not improve and increased overfitting risk
```

My conclusion: I found PCA-128 to be the best practical compression level. It retained enough signal while removing many 
noisy directions.

### Stage 5: compact trajectory/geometric features

The largest conceptual improvement came after I added compact trajectory features. Instead of only asking “what is the last 
hidden vector?”, I described how representations behave over the last tokens and across layers.

This was useful because hallucination may appear as a representation-dynamics pattern: instability, unusual drift, different
covariance structure, or mismatch between the final token and local answer-window mean.

The compact trajectory features improved accuracy over the raw-only setup, so I included them in the final feature extractor.

My conclusion: compact geometry helped; simply adding more raw dimensions did not.

### Stage 6: expanded INSIDE/EigenScore-like spectral features

I then tried to expand the spectral feature block. My expanded version included many extra covariance-spectrum statistics: 
stable logdet, Frobenius norm, condition ratio, top-k eigenvalue ratios, inter-layer spectral features, and more.

This made performance worse:

```text
compact trajectory / exp18-style features: better accuracy
expanded spectral features: lower accuracy and lower stability
```

My interpretation is that the compact feature block already captured the useful spectral information, while the extended 
block added noisy, unstable scalar dimensions.

My conclusion: spectral ideas were useful only in a small, regularized form. I discarded the expanded version.


### Stage 7: layer search

After stabilizing the probe, I searched over raw hidden layers.

Single raw-layer experiments showed that layers around `-8` and `-12` were stronger than the final layer. The best single 
layer by AUROC in my runs was `[-12]`, but the best accuracy came from a multi-layer combination:

```python
RAW_LAYERS = [-4, -8, -12, -16]
```

This combination gave me a better balance than using very late layers only or too many adjacent layers. Runs such as 
`[-6, -8, -10, -12]`, `[-8, -10, -12, -14]`, and `[-10, -12, -14, -16]` did not improve accuracy.

My conclusion: the useful signal is in intermediate/deeper layers, but the exact spacing matters. My best raw-layer 
combination was `[-4, -8, -12, -16]`.


### Stage 8: trajectory layer choice

Other trajectory choices were worse or unstable in my runs. For example, using `[-4, -8, -12, -16, -20]` sometimes gave a 
reasonable result, but not as consistently as `[-8, -10, -12, -14, -16]`.

My conclusion: the final trajectory block should focus on intermediate/deeper layers around `-8` to `-16`.

### Final probe choice: logistic regression

The exploratory stages above motivated a low-capacity decision rule on top of high-dimensional features. The submitted 
pipeline therefore uses StandardScaler → PCA(128) → LogisticRegressionCV (L2, Cs=10, inner cv=5) instead of an MLP, for the 
reasons stated in Section 3.

I stopped at the configuration with approximately 74.16% 5-fold accuracy because it is the best result by the target metric, 
as reflected in the 5-fold metrics in `results.json`:

```text
baseline accuracy:      70.10%
final CV accuracy:      74.16%
absolute improvement:   +4.06 percentage points
```

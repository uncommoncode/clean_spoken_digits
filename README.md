# What is the Clean Spoken Digits Dataset?

Clean Spoken Digits (CSD) is a dataset of high quality clean synthetic speakers saying the words zero through nine. Like MNIST, it is a valuable dataset for rapid iteration and testing. Also like MNIST, you probably need a more sophisticated dataset and training pipeline for anything production oriented.

Open source audio recordings are plauged by many challenges not seen in this dataset including:

 * Additive noises coming from the recording environment or electrical noise of the recording setup, or even audience applause in TED talks.
 * Convolutional noise from reverberation, frequency response of migrophones, and gain levels across a variety of setups from recording studios like in VCTK to whatever may go in LibriSpeech.
 * Compression artifacts or resampling rolloff errors from postprocessing.
 * DC offsets from bad recording setups.
 * Transient discontinuities that sound like crickets and other artifacts.
 * Inconsistent trimming between clips. Some clips may be empty while others have partial words.
 * Bad labels. I've seen clips where the audio file is empty but labeled as speech.

 All of these factors add up to many public audio datasets have significant amount of noise that makes it difficult to understand performance differences between models and debug training pipelines.


 # What does it contain?

 70000 juicy 16khz single channel wav files split across 6000 for training and 1200 for test with balanced genders.

 These are generated from high quality Wavenet speech synthesis models with random parameters for:

  * Dialect: `en-US`, `en-GB`, and `en-AU`
  * Volume: 0-6 dB gain
  * Pitch: -6 to 6
  * Speaking Rate: 0.85 to 1.35x
  * Inflection: neutral (e.g. `'one'`), question (e.g. `one?`), exclaimation (e.g. `'one!'`), definitive (e.g. `'one.'`)

We constrained the parameters to produce plausibly human-like voices. We found the `en-IN` voices to have unrealistic distortion with variation in pitch and speaking rate so they are not available at the moment.

Due to the low number of voice names, we have a combined set of 3 voices held out in the test set and 3 voices from the training set. However due to the random pitch, volume, speaking rate, and inflection the synthetic speech is still different between training and test.

## Training Distribution
| Gender | Count | 
|--------|-------| 
| female | 3000  | 
| male   | 3000  | 

| Voice           | Count | 
|-----------------|-------| 
| en-US-Wavenet-E | 500   | 
| en-AU-Wavenet-C | 500   | 
| en-US-Wavenet-D | 500   | 
| en-AU-Wavenet-A | 500   | 
| en-US-Wavenet-B | 500   | 
| en-GB-Wavenet-F | 500   | 
| en-GB-Wavenet-D | 500   | 
| en-GB-Wavenet-C | 500   | 
| en-GB-Wavenet-B | 500   | 
| en-US-Wavenet-A | 500   | 
| en-AU-Wavenet-B | 500   | 
| en-US-Wavenet-C | 500   | 


## Test Distribution
| Gender | Count | 
|--------|-------| 
| female | 600   | 
| male   | 600   | 

| Voice            | Count | 
|------------------|-------| 
| en-US-Wavenet-F  | 200   | 
| en-GB-Wavenet-A  | 200   | 
| en-AU-Wavenet-D  | 200   | 
| en-US-Wavenet-A* | 200   | 
| en-US-Wavenet-B* | 200   | 
| en-AU-Wavenet-C* | 200   | 

*These voice names are not unique to the test set.


# Want to get going quickly?

The most common input to audio models is a melspectrogram. We have preprocessed features to a small size for efficient processing with 8khz downsampled 8-mel and 32-mel bin targets, with zero padding with random centering. Because the clips are perfectly clean the zero padding should not otherwise confuse a model.

Features are stored in the `features` field, where it is typical to do a `log(x + 1)` transformation on the mel inputs.

You will want to choose your label (probably the word but you could also try to classify the suffix, gender, voice identity, speed, etc.) in the `labels` field.

## Keras

An example training in Keras would look like:

```python
dataset = np.load('clean_spoken_digits_mel32.npz', allow_pickle=True)
X_train = dataset['train_features'].astype(np.float32)
y_train = np.array([label['word_id'] for label in dataset['train_labels']])

X_test = dataset['test_features'].astype(np.float32)
y_test = np.array([label['word_id'] for label in dataset['test_labels']])

X_train = np.log1p(X_train)
y_train = keras.utils.to_categorical(y_train)

X_test = np.log1p(X_test)
y_test = keras.utils.to_categorical(y_test)

model = keras.models.Sequential([
  keras.layers.Input((None, 32)),
  keras.layers.GRU(32),
  keras.layers.Dense(10, activation='softmax'),
])
model.build()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit(X_train, y_train)

loss = model.evaluate(X_test, y_test)
```

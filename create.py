import copy
import csv
import hashlib
import itertools
import multiprocessing
import pathlib
import random
import shutil
import tempfile

import librosa
import numpy as np
import soundfile as sf
import tqdm
from google.cloud import texttospeech

N_TRAIN_SPEAKERS = 600
N_TEST_SPEAKERS = 120

OUT_DIR = "clean_spoken_digits"

MALE = "male"
FEMALE = "female"

GENDER_NAMES = {
    "male": texttospeech.SsmlVoiceGender.MALE,
    "female": texttospeech.SsmlVoiceGender.FEMALE,
}

VOICES = {
    "en-US": {
        "en-US-Wavenet-C": FEMALE,
        "en-US-Wavenet-E": FEMALE,
        "en-US-Wavenet-F": FEMALE,
        "en-US-Wavenet-A": MALE,
        "en-US-Wavenet-B": MALE,
        "en-US-Wavenet-D": MALE,
    },
    "en-GB": {
        "en-GB-Wavenet-A": FEMALE,
        "en-GB-Wavenet-C": FEMALE,
        "en-GB-Wavenet-F": FEMALE,
        "en-GB-Wavenet-B": MALE,
        "en-GB-Wavenet-D": MALE,
    },
    "en-AU": {
        "en-AU-Wavenet-A": FEMALE,
        "en-AU-Wavenet-C": FEMALE,
        "en-AU-Wavenet-B": MALE,
        "en-AU-Wavenet-D": MALE,
    },
}

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
WORD_IDS = {word: index for index, word in enumerate(WORDS)}


def synthesize_text(
    text, volume_gain_db, speaking_rate, pitch, language_code, name, gender
):
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, name=name, ssml_gender=gender,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        volume_gain_db=volume_gain_db,
        speaking_rate=speaking_rate,
        pitch=pitch,
        sample_rate_hertz=16000,
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    return response.audio_content


def generate_random_config(n_speakers, split, voice_list, seed, existing_speaker_count):
    speaking_rate_range = [0.85, 1.35]
    pitch_range = [-6, 6]
    volume_gain_db_range = [0, 6]
    suffixes = ["", "?", "!", "."]

    random.seed(seed)

    voice_list_iter = itertools.cycle(voice_list)

    for i in range(n_speakers):
        speaker_id = i + existing_speaker_count
        language_code, voice = next(voice_list_iter)
        gender = VOICES[language_code][voice]
        speaking_rate = random.uniform(*speaking_rate_range)
        pitch = random.uniform(*pitch_range)
        volume_gain_db = random.uniform(*volume_gain_db_range)
        suffix = random.choice(suffixes)
        for word in WORDS:
            path = f"{split}/{word}_{speaker_id}_{gender[0]}.wav"
            word_id = WORD_IDS[word]
            sample = {
                "speaking_rate": speaking_rate,
                "pitch": pitch,
                "volume_gain_db": volume_gain_db,
                "language_code": language_code,
                "voice": voice,
                "gender": gender,
                "word": word,
                "word_id": word_id,
                "suffix": suffix,
                "text": word + suffix,
                "speaker_id": speaker_id,
                "split": split,
                "path": path,
            }
            yield sample


def create_audio(config):
    path = config["path"]
    out_path = f"{OUT_DIR}/{path}"
    if pathlib.Path(out_path).exists():
        return config
    gender = GENDER_NAMES[config["gender"]]
    audio = synthesize_text(
        text=config["text"],
        volume_gain_db=config["volume_gain_db"],
        speaking_rate=config["speaking_rate"],
        pitch=config["pitch"],
        language_code=config["language_code"],
        name=config["voice"],
        gender=gender,
    )
    with tempfile.NamedTemporaryFile("wb") as w:
        w.write(audio)
        shutil.copyfile(w.name, out_path)
    return config


def create_dataset_csv():
    pathlib.Path(f"{OUT_DIR}/train").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{OUT_DIR}/test").mkdir(parents=True, exist_ok=True)

    shutil.copyfile("DATA_LICENSE.txt", f"{OUT_DIR}/LICENSE.txt")

    if pathlib.Path(f"{OUT_DIR}/labels.csv").exists():
        print("Already have partial dataset! Skipping creating new label.")
        return

    random.seed(413)
    train_voice_list = []
    test_voice_list = []
    for lang, value in VOICES.items():
        voices = [(lang, voice) for voice in value.keys()]
        random.shuffle(voices)
        test_voice_list.append(voices[0])
        train_voice_list += voices[1:]

    # Shuffle here to prevent any bias to the end of the list vs the beginning.
    random.shuffle(train_voice_list)
    # Double the number of test voices with half of the voice ids from training.
    n = len(test_voice_list)
    test_voice_list += train_voice_list[:n]
    random.shuffle(test_voice_list)

    seed_a = random.randint(1, 2 ** 31 - 1)
    train_configs = list(
        generate_random_config(
            n_speakers=N_TRAIN_SPEAKERS,
            split="train",
            seed=seed_a,
            voice_list=train_voice_list,
            existing_speaker_count=0,
        )
    )
    total_speakers = N_TRAIN_SPEAKERS
    seed_b = random.randint(1, 2 ** 31 - 1)
    test_configs = list(
        generate_random_config(
            n_speakers=N_TEST_SPEAKERS,
            split="test",
            seed=seed_b,
            voice_list=test_voice_list,
            existing_speaker_count=total_speakers,
        )
    )
    total_speakers += N_TEST_SPEAKERS

    print(f'Generating dataset with {total_speakers} speakers and {len(WORDS)} utterances each.')

    configs = train_configs + test_configs

    with open(f"{OUT_DIR}/labels.csv", "w", newline="") as csvfile:
        fieldnames = configs[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(configs)


def read_label_csv(path):
    with open(path, "r", newline="") as r:
        reader = csv.DictReader(r)
        configs = [config for config in reader]
    for config in configs:
        for key in ["speaking_rate", "pitch", "volume_gain_db"]:
            config[key] = float(config[key])
        for key in ["speaker_id"]:
            config[key] = int(config[key])
    return configs


def create_dataset_wavs():
    configs = read_label_csv(f"{OUT_DIR}/labels.csv")

    for config in tqdm.tqdm(configs):
        create_audio(config)


def create_dataset_npzs():
    configs = read_label_csv(f"{OUT_DIR}/labels.csv")

    random.seed(177)
    random.shuffle(configs)

    sample_rate = 8000

    labels = []
    features_8 = []
    features_32 = []

    print("Computing spectrograms...")
    for config in tqdm.tqdm(configs):
        path = config["path"]
        out_path = f"{OUT_DIR}/{path}"

        with open(out_path, "rb") as r:
            wav, sr = sf.read(r, always_2d=True)

        wav = wav[:, 0]
        if sr != sample_rate:
            wav = librosa.resample(wav, sr, sample_rate)

        labels.append(config)

        f8 = librosa.feature.melspectrogram(
            wav, n_fft=512, hop_length=128, sr=sample_rate, n_mels=8
        )
        features_8.append(f8.T)

        f32 = librosa.feature.melspectrogram(
            wav, n_fft=512, hop_length=128, sr=sample_rate, n_mels=32
        )
        features_32.append(f32.T)

    print("Padding...")
    # Pad to same dimensions
    # TODO: find outlier!
    max_frames = max(f.shape[0] for f in features_8)

    for i in range(len(features_8)):
        config = configs[i]
        key = (
            config["path"]
            + config["voice"]
            + config["language_code"]
            + f'{config["speaking_rate"]}{config["pitch"]}{config["volume_gain_db"]}{config["suffix"]}'
        )
        sha1 = hashlib.sha1(key.encode()).hexdigest()
        stable_random_int = int(sha1, 16)
        feature_8 = features_8[i]
        feature_32 = features_32[i]
        padding = max_frames - feature_8.shape[0]
        pad_start = stable_random_int % padding if padding != 0 else 0
        pad_end = padding - pad_start
        features_8[i] = np.pad(
            feature_8, ((pad_start, pad_end), (0, 0)), mode="constant"
        )
        features_32[i] = np.pad(
            feature_32, ((pad_start, pad_end), (0, 0)), mode="constant"
        )

    # Convert to fp16 to save disk space. The average and maximum error is small, particularly after a log1p.
    features_8 = np.array(features_8).astype(np.float16)
    features_32 = np.array(features_32).astype(np.float16)

    train_mask = np.array([label["split"] == "train" for label in labels])
    labels = np.array(labels)

    print("Saving features...")
    np.savez_compressed(
        "clean_spoken_digits_mel8.npz",
        train_features=features_8[train_mask],
        train_labels=labels[train_mask],
        test_features=features_8[~train_mask],
        test_labels=labels[~train_mask],
    )
    np.savez_compressed(
        "clean_spoken_digits_mel32.npz",
        train_features=features_32[train_mask],
        train_labels=labels[train_mask],
        test_features=features_32[~train_mask],
        test_labels=labels[~train_mask],
    )

    print("Done!")


if __name__ == "__main__":
    #create_dataset_csv()
    #create_dataset_wavs()
    create_dataset_npzs()

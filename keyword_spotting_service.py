import os

import numpy as np
import tensorflow.keras as keras
import librosa
from VoiceRecorder import voice_recorder

MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050  # 1 sec


class _Keyword_Spotting_service:
    model = None
    _mappings = [
        "arbaa",
        "tlete",
        "tnen",
        "wahad"
    ]
    _instance = None  # implementing this class as a singleton

    def predict(self, file_path):
        # extract the MFCCs
        MFCCs = self.preprocess(file_path)  # (# segments, # coefficients)
        # convert 2D MFCCs array into 4D array -> (# samples, # segments, # coefficients, # channels = 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        # make prediction
        predictions = self.model.predict(MFCCs)  # [[probabilities]]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        # load audio file
        signal, sr = librosa.load(file_path)
        # ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]
        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft,
                                     hop_length=hop_length)
        return MFCCs.T


# factory function responsible for instantiating a keyword spotting service
def Keyword_Spotting_Service():
    # ensure that we only have one instance of KSS
    if _Keyword_Spotting_service._instance is None:
        _Keyword_Spotting_service._instance = _Keyword_Spotting_service()
        _Keyword_Spotting_service.model = keras.models.load_model(MODEL_PATH)

    # we return the instance anyway insuring that it is instantiated once
    return _Keyword_Spotting_service._instance


if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    kw1 = kss.predict("test\\Arbaa.wav")
    kw2 = kss.predict("test\\Tlete.wav")
    kw3 = kss.predict("test\\Tnen.wav")
    kw4 = kss.predict("test\\Wahad.wav")
    print(f"Predicted kw: {kw1}, {kw2}, {kw3}, {kw4}")

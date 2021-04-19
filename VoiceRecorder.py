import sounddevice as sd
from scipy.io.wavfile import write


def voice_recorder():

    print("Recording has started")

    # Sampling frequency
    freq = 22050

    # Recording duration
    duration = 4

    # Start recorder with the given values
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq),
                       samplerate=freq, channels=2)

    # Record audio for the given number of seconds
    sd.wait()

    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write("recording0.wav", freq, recording)

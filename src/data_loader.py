import numpy as np
import librosa
import math

SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# def extract_melspectrogram(signal, sample_rate, n_mels=128, n_fft=2048, hop_length=512):
#     mel_spec = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=n_fft,
#                                                hop_length=hop_length, n_mels=n_mels)
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#     return mel_spec_db.T[..., np.newaxis]  # Shape: (time, mel, 1)



# def preprocess_audio(file_path, num_segments=10):
#     signal, sr = librosa.load(file_path, sr=22050)
#     duration = librosa.get_duration(y=signal, sr=sr)

#     samples_per_segment = int(sr * duration / num_segments)

#     segment_features = []
#     for d in range(num_segments):
#         start = d * samples_per_segment
#         end = start + samples_per_segment
#         segment = signal[start:end]

#         if len(segment) == samples_per_segment:
#             mel = extract_melspectrogram(segment, sr)
#             if mel.shape[1] == 128:  # flexible check
#                 segment_features.append(mel)

#     return np.array(segment_features)

def extract_melspectrogram(signal, sample_rate, n_mels=128, n_fft=2048, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=n_fft,
                                               hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db.T[..., np.newaxis]  # Shape: (time, mel, 1)


def extract_features(file_path, n_fft=2048, hop_length=512, num_segments=5):
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    feat = []
    y, sr = librosa.load(file_path, duration=30)  # Load 30 seconds of audio
    for d in range(num_segments):
        start = samples_per_segment * d
        end = start + samples_per_segment
        segment = y[start:end]
        mfcc_feat = extract_melspectrogram(segment, sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)
        if len(mfcc_feat) == num_mfcc_vectors_per_segment:
            # mfcc_feat = mfcc_feat.reshape((num_mfcc_vectors_per_segment, 128, 1))
            feat.append(mfcc_feat)
            print(f"{file_path}, segment {d+1} ✅")
    print(np.array(feat).shape)                
    return np.array(feat)
    
    # mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,  n_fft=2048, hop_length=512)
    # mel_db = librosa.power_to_db(mel, ref=np.max)

    # # Make sure it has 259 time steps
    # if mel_db.shape[1] < 259:
    #     pad_width = 259 - mel_db.shape[1]
    #     mel_db = np.pad(mel_db, pad_width=((0,0), (0, pad_width)), mode='constant')
    # else:
    #     mel_db = mel_db[:, :259]  # Trim if it's longer

    # mel_db = mel_db.T  # Transpose to (259, 128)
    # mel_db = np.expand_dims(mel_db, axis=[0, -1])  # Final shape: (1, 259, 128, 1)

    # return mel_db

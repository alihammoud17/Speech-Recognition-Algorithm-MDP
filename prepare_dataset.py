import os
import json
import librosa

# Constants
# Dataset used for training
DATASET_PATH = "dataset"
# Where the data is stored
JSON_PATH = "data.json"
# Number of samples considered to preprocess data
SAMPLES_TO_CONSIDER = 22050  # 1 sec worth of sound


# Main function to preprocess the data
def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    # create data dictionary
    data = {
        "mappings": [],  # keywords
        "labels": [],  # a value for each audio file in the dataset
        "MFCCs": [],  # MFCC for each audio file
        "files": []  # filenames with path for each audio file
    }
    # loop through all the sub-dirs
    # walk through a folder structure recursively top-down
    for i, (dir_path, dir_names, filenames) in enumerate(os.walk(dataset_path)):
        # we need to ensure that we are not at root level
        if dir_path is not dataset_path:
            # update mappings
            category = dir_path.split("\\")[-1]  # category name ex: dataset\\wahad -> [dataset, wahad]
            data["mappings"].append(category)
            print(f"Processing {category}")

            # loop through filenames and extract MFCCs
            for f in filenames:
                # get file path
                file_path = os.path.join(dir_path, f)  # gives us the whole file path

                # load audio file
                signal, sr = librosa.load(file_path)

                # ensure the audio file is at least 1 second
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # enforce on 1 sec. long signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract the MFCCs
                    MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc,
                                                 hop_length=hop_length, n_fft=n_fft)

                    # store data
                    data["labels"].append(i - 1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path}: {i - 1}")

    # store in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)

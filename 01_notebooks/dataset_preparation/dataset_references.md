# Dataset References

This file summarizes the external datasets used or considered in the project and points to the notebooks where their download and preprocessing logic is implemented.

## Word-Level Sign Language Recognition

| Dataset | Purpose | Notes |
|---|---|---|
| WLASL | Main word-level ASL recognition dataset | Used for WLASL30, WLASL100, WLASL300, WLASL1000 and larger scaling experiments. |
| MS-ASL | External word-level ASL dataset | Candidate dataset for cross-dataset comparison and signer-independent evaluation. |

Relevant notebooks:

- `01_notebooks/word_recognition/14_wlasl2000_video_audit.ipynb`
- `01_notebooks/word_recognition/15_wlasl2000_landmark_extraction.ipynb`
- `01_notebooks/word_recognition/16_dataset_subsets.ipynb`
- `01_notebooks/word_recognition/19_msasl_download.ipynb`
- `01_notebooks/word_recognition/20_msasl_landmark_extraction.ipynb`

## Earlier Hand Detection Work

| Dataset | Purpose | Notes |
|---|---|---|
| EgoHands | Hand detection baseline | Used for the earlier bounding-box detection phase. |
| Kaggle Hand Dataset | Hand detection baseline | Combined with EgoHands for a more varied hand detector training set. |

Relevant notebooks:

- `01_notebooks/hand_detection/01_YOLO_HandDetector_MixedDataset.ipynb`
- `01_notebooks/hand_detection/02_SSD_FasterRCNN_Comparison.ipynb`

## Earlier Letter-Level Work

| Dataset | Purpose | Notes |
|---|---|---|
| Sign Language MNIST | Letter-level ASL baseline | Used only in the early letter-classification phase. |
| ASL Alphabet | Letter-level ASL baseline | Used only in the early letter-classification phase. |

Relevant notebook:

- `01_notebooks/dataset_preparation/01_Prepare_ASL_Dataset_Combined.ipynb`

## Kaggle Credentials

Kaggle downloads require a local `utils/kaggle.json` file with the standard Kaggle API format:

```json
{
  "username": "kaggle_username",
  "key": "kaggle_api_key"
}
```

The credentials file is ignored by Git and must not be committed.

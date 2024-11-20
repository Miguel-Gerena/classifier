# What Was That?: Lip-Reading Silent Videos

## Authors: Akayla Hackson, Miguel Gerena, Linyin Lyu
**Stanford University**

**Contact**: [akayla@stanford.edu](mailto:akayla@stanford.edu), [miguelg2@stanford.edu](mailto:miguelg2@stanford.edu), [llyu@stanford.edu](mailto:llyu@stanford.edu)

---

### Abstract
This study develops three models to advance lip-reading technology using silent video analysis, achieving different levels of success in recognizing spoken words and sentences purely from visual information.

---

### Introduction
Lip-reading, or visual speech recognition, involves transcribing text from silent videos. It has applications in diverse fields such as surveillance, silent video conferencing, and archival film transcription. The task faces challenges due to visual ambiguities among similar phonetic sounds and diverse mouth movements across speakers.

---

### Model Development
- **Speaking Detection Model**: Uses a 3D-CNN to detect speaking activity with 77.63% accuracy.
- **Word Prediction Model (LRW)**: Extends the 3D-CNN with LSTM to handle temporal dependencies, focusing on single-word prediction with 52.19% accuracy.
- **Sentence Prediction Model**: Combines the previous architectures with an Encoder-Decoder Transformer layer aimed at constructing full sentences, which needs further improvement due to issues with local minima.

---

### Methodology
Our approach combines spatiotemporal CNNs for feature extraction from video frames and Bi-LSTM for modeling temporal sequence dynamics. The models are further enhanced by transformer layers for handling longer-range dependencies and improving sentence structure prediction.

---

### Datasets and Preprocessing
- **LRS2 Dataset**: Used for predicting both speaking status and full sentences.
- **LRW Dataset**: Employed for single-word predictions.
- Videos were processed to maintain uniform frame rates and normalized for consistent input to the models.

---

### Experiments and Results
Initial experiments on speaking detection provided a foundation with reasonable accuracy. The single-word prediction model showed promise but highlighted the complexity of extending to full sentences. The most advanced model, intended for full sentences, demonstrated the need for significant refinement.

---

### Conclusions and Future Work
While the models for detecting speech and predicting words performed commendably, the sentence prediction model requires further development. Future work will focus on refining the models using additional data, extended training, and potentially incorporating more advanced techniques like self-supervised learning to handle the complexities of lip reading.

---

### Acknowledgements
We thank our colleagues and department for their support and the insightful discussions that helped shape this research.

---

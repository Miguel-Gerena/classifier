# Patent Acceptance Prediction With Large Language Models

## Stanford CS224N Custom Project
**Project proposed & mentored by Mirac Suzgun**

**Authors**: Akayla Hackson, Miguel Gerena

**Affiliations**:
- Department of Electrical Engineering, Stanford University
- Department of Computer Science, Stanford University

**Contact**: [akayla@stanford.edu](mailto:akayla@stanford.edu), [miguelg2@stanford.edu](mailto:miguelg2@stanford.edu)

---

### Abstract
This paper presents a Language Model (LM) developed to predict patent approval outcomes for small businesses and inventors. Leveraging the linguistic analysis of patent application texts, our model achieved a new state-of-the-art (SOTA) accuracy of 64.37%, surpassing the previous best model's accuracy of 57.96%. Despite this achievement, our model has reached a performance plateau and requires further enhancement.

---

### Introduction
Small businesses and inventors face significant challenges in securing patent approvals, often hindered by the complex patent application process and the need for precise technical documentation. Our LM aims to democratize the patent application process by providing predictive insights into the viability of patent applications, thereby helping to optimize their chances of approval.

---

### Model Development
- **Initial Model**: Our baseline model, a fine-tuned variant of BERT, exceeded the current best model with 64.37% accuracy.
- **Advanced Models**: We experimented with larger and more capable models such as Mistral-7b and Gemma-7b, incorporating advanced techniques like LoRA and model quantization to enhance performance further.

---

### Methodology
We employed a variety of LMs and techniques to refine our prediction model:
- **BERT and Variants**: Used for their powerful contextual understanding capabilities.
- **Efficiency Techniques**: Implementation of LoRA and model quantization to reduce computational demands while maintaining high performance.

---

### Experiments and Results
Our experiments utilized the Harvard USPTO Patent Dataset, focusing on patent acceptance prediction. Despite initial successes, further improvements are necessary to overcome the performance plateau encountered by our current best models.

---

### Conclusion and Future Work
While we have surpassed the existing SOTA models in accuracy, our journey continues as we seek to break through the current performance limitations. Future efforts will focus on exploring more sophisticated models and further refining our approaches to better handle the complexities of patent applications.

---

### Acknowledgements
We express our gratitude to our mentor Mirac Suzgun and all who supported this project.

---

For more information or to contribute to this project, please contact the authors at the provided email addresses.


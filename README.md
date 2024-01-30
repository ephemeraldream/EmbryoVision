# High-Precision Embryological Image Analysis with Deep Learning

### A complete Web Aplication for EmbryoImage Recognition.

## Introduction

In the realm of contemporary medical Data Science, the project at hand stands at the forefront of technological innovation. We aim to build a high-precision Multi-Object Detection and Classification Neural Network (MODCNN) that detects and classifies numerous holes within a cascade of embryological images. This endeavor combines a classical convolutional base with multitasking capabilities and explores the integration of the latest YOLOv7 model alongside transformer-based pre-trained models.

Embryological labs, inundated with image data, stand to benefit immensely from our automated detection and classification system. By streamlining the analysis process, we empower embryologists to direct their expertise where it matters most, enhancing patient outcomes and advancing medical imaging.

![Embryological Image](https://github.com/ephemeraldream/EmbryoVision/blob/main/utils/photo_2024-01-29_23-29-05.jpg)

## Project Overview

Our proprietary dataset, sourced directly from a Russian embryological lab, presents a unique challenge due to its substantial size of approximately 8 TB. We tackle this by employing rigorous data compression, pre-processing, and augmentation techniques, preparing the dataset for optimal neural network training.

The MODCNN's design is tailored to embryological image analysis, diverging into specialized branches for detection and classification. We also experiment with YOLOv7's real-time object detection and the potential of transformer-based models known for capturing complex data patterns.

![Embryological Image](https://github.com/ephemeraldream/EmbryoVision/blob/main/utils/photo_2024-01-29_23-29-05.jpg)

## Materials and Methods

Development takes place in the laboratory of LLC "WESTTRADE LTD". The raw data collection was conducted in the ART (Assisted Reproductive Technologies) laboratory of the "Family" medical center (Ufa, Russia). The visual information on the preimplantation development of human embryos up to the blastocyst stage (0–6 days post-insemination) was obtained using the "EmbryoViewer" incubator with a time-lapse video recording system (LLC "WESTTRADE LTD", Russia). Embryo cultivation was performed individually in special WOW dish microwells (Vitrolife, Sweden and VivaVitro, China). The dataset was labeled using Label Studio Community Edition software. A recurrent convolutional neural network architecture was selected, and the model was trained on numerous images.

![Embryological Image](https://github.com/ephemeraldream/EmbryoVision/blob/main/utils/photo_2024-01-29_23-29-05.jpg)

## Discussion

Modern technologies enabling decision support systems in various medical fields are opening new horizons in assisted reproductive technologies, particularly in embryology. These technological solutions allow for manual or automatic construction of a human embryo's morphodynamic profile, a chronological vector of morphokinetic states based on time-lapse image series.

The prototype system for comparative assessment of implantation potential was developed in the laboratory of LLC "WESTTRADE LTD" based on the outcomes of transferring 419 embryos cultivated in "EmbryoViewer" dry planar incubators with a ready-made gas mixture containing 6% CO2, 5% O2.

## Technologies Used

- **PyTorch**: For building and training state-of-the-art neural networks.
- **Django + Svelte JS**: For deploying the web application that interfaces with the neural network models.

## Contact

For further inquiries or collaboration opportunities, feel free to reach out:

- Telegram: [@immanentdream](https://t.me/immanentdream)
- Email: [akinfiev@arizona.edu](mailto:akinfiev@arizona.edu)

---

*This project is a testament to the power of deep learning in enhancing the quality and efficiency of medical diagnostics and research.*




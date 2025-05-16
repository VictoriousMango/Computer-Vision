# Computer Vision Learning Repository

## Motivation
This repository is dedicated to learning and implementing Computer Vision concepts from class notes. It serves as a structured resource for understanding theoretical foundations, practical implementations, and real-world applications of Computer Vision. The goal is to consolidate knowledge, provide hands-on coding examples, and explore topics like camera calibration, deep learning, edge detection, and more. This repository is ideal for students, researchers, or enthusiasts aiming to deepen their understanding of Computer Vision through organized notes and implementations.

## Folder Structure
The repository is organized into folders, each corresponding to a specific topic from the class notes. Below is the structure and a brief overview of each folder's contents:

```
computer-vision-learning/
│
├── Camera_Calibration/
├── Deep_Learning/
├── Edge_Detection/
├── Exams_and_Quizzes/
├── Estimating_Homography/
├── Feature_Extraction/
├── Harris_Corner_and_RANSAC/
├── HOG_and_BoVW/
├── Image_Processing/
├── Image_Segmentation/
├── Knowledge_Engineering/
├── Introduction_to_CV/
├── Optical_Flow_Estimation/
├── Stereo_Vision/
├── Projective_Geometry/
└── README.md
```

## Folder Contents

### Camera_Calibration
- **Focus**: Camera calibration techniques and pose estimation.
- **Contents**:
  - Implementation of Zhang’s method using 2D checkerboard patterns.
  - Direct Linear Transformation (DLT) for calibration and pose estimation.
  - Computing intrinsic (K) and extrinsic parameters via matrix decomposition (Cholesky, SVD).
  - P3P (Perspective-3-Point) algorithm for camera localization.
  - Mathematical operations like matrix-vector products and constraint exploitation.
  - Applications in robotics.

### Deep_Learning
- **Focus**: Fundamentals of deep learning for Computer Vision.
- **Contents**:
  - Overview of machine learning (supervised, unsupervised, reinforcement learning).
  - Neural network components (neurons, layers, activation functions like sigmoid, softmax).
  - Training techniques: gradient descent, loss functions (cross-entropy), regularization, k-fold cross-validation.
  - Architectures: CNNs, RNNs (including LSTMs), Autoencoders, VAEs, and GANs.
  - Projects: Convective storm detection, depth estimation from RGB images, object detection, and more.
  - Example implementations (e.g., MNIST digit recognition).

### Edge_Detection
- **Focus**: Techniques for detecting object boundaries.
- **Contents**:
  - Implementation of Sobel and Canny edge detectors.
  - Hough Transform for detecting lines and shapes (including polar coordinates).
  - Template matching using convolution and cross-correlation.
  - Smoothing techniques (Gaussian, bilateral, NL-means filters).
  - Integral image for rapid area calculations.
  - Color image transformations (RGB, CMY).

### Exams_and_Quizzes
- **Focus**: Practice questions and solutions covering Computer Vision concepts.
- **Contents**:
  - Problems on pinhole cameras, affine transforms, and vanishing points.
  - Mean-Shift and K-Means clustering for segmentation.
  - Harris feature detector and RANSAC for model fitting.
  - Optical flow (Lucas-Kanade, Horn-Schunck) and aperture problem.
  - Gaussian filters, image sharpening, and Hough Transform applications.

### Estimating_Homography
- **Focus**: Homography estimation for plane-to-plane mapping.
- **Contents**:
  - Homography definition and equations (8 DoF, 4 point correspondences).
  - Solving homography using SVD.
  - Applications in image mosaicing.
  - Exploration of projective homography and cross-ratio invariance.

### Feature_Extraction
- **Focus**: Image processing and feature extraction techniques.
- **Contents**:
  - Image sampling, quantization, and histogram analysis.
  - Linear systems and filtering (convolution, correlation).
  - Implementation of discrete convolution and cross-correlation.
  - Use cases for histograms and image functions.

### Harris_Corner_and_RANSAC
- **Focus**: Feature detection and robust model fitting.
- **Contents**:
  - Harris corner detector for keypoint localization.
  - RANSAC for line fitting and translation estimation.
  - Local invariant feature detection and matching workflows.
  - Implementations for detecting repeatable and precise keypoints.

### HOG_and_BoVW
- **Focus**: Advanced feature extraction and image representation.
- **Contents**:
  - Histogram of Oriented Gradients (HOG) for edge-based features.
  - Bag of Visual Words (BoVW) for image classification.
  - K-Means clustering for dictionary learning.
  - TF-IDF reweighting for histogram comparisons.
  - SIFT keypoint matching recap.

### Image_Processing
- **Focus**: Core image processing techniques.
- **Contents**:
  - Image transformations (filtering, warping).
  - Template matching using SSD and normalized cross-correlation.
  - Frequency domain analysis via Fourier transform.
  - Implementation of point and neighborhood operations.

### Image_Segmentation
- **Focus**: Partitioning images into meaningful regions.
- **Contents**:
  - Thresholding (Otsu’s method) and image binarization.
  - Region-based methods (region merging, RAG, quadtree).
  - Clustering-based segmentation (K-Means, Mean Shift).
  - Applications in medical imaging, object detection, and 3D reconstruction.

### Knowledge_Engineering
- **Focus**: Integrating knowledge-based systems with Computer Vision.
- **Contents**:
  - Knowledge representation (RDF, OWL, ontologies).
  - Semantic web technologies for KRR.
  - Ontology development for Computer Vision (e.g., COCOnet, ImageNet KnowledgeGraph).
  - Use cases: fine-grained recognition, contextual understanding, semantic search.
  - Challenges in bridging the semantic gap.

### Introduction_to_CV
- **Focus**: Overview of Computer Vision.
- **Contents**:
  - Course logistics and prerequisites (Python, TensorFlow/PyTorch, linear algebra).
  - Classical vs. modern Computer Vision pipelines.
  - Historical context and challenges in Computer Vision.
  - Theoretical and practical concepts for research.

### Optical_Flow_Estimation
- **Focus**: Motion estimation in image sequences.
- **Contents**:
  - Brightness constancy equation and assumptions.
  - Lucas-Kanade method for local optical flow estimation.
  - Implementation of partial derivative computations.
  - Exploration of global vs. local flow methods.

### Stereo_Vision
- **Focus**: Depth perception using stereo cameras.
- **Contents**:
  - Epipolar geometry and fundamental matrix estimation (8-point algorithm).
  - Stereo correspondence and matching algorithms.
  - Depth and disparity relationships.
  - Applications in medical imaging and 3D reconstruction.

### Projective_Geometry
- **Focus**: Mathematical foundations for perspective projections.
- **Contents**:
  - Homogeneous coordinates and 2D line arithmetic.
  - Conics and 3D geometry (planes, lines, quadrics).
  - 2D transformations and their applications in multi-view geometry.
  - Implementations for camera calibration.

---

This repository will evolve as implementations are added to each folder. Contributions, suggestions, or additional resources are welcome!
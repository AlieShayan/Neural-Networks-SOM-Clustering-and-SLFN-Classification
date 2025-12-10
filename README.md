# Neural Networks: SOM Clustering and SLFN Classification

This repository contains implementations of **Self-Organizing Maps (SOM)** for clustering and classification, along with a **Single Layer Feedforward Network (SLFN)** built entirely from scratch. The project demonstrates unsupervised and supervised learning techniques applied to real-world datasets.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-blue?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-blue?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-blue?logo=matplotlib&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-orange?logo=scikit-learn&logoColor=white)
![MiniSom](https://img.shields.io/badge/MiniSom-Library-green)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project explores two fundamental neural network architectures:

1. **Self-Organizing Maps (SOM)**: A type of artificial neural network trained using unsupervised learning to produce a low-dimensional representation of input space
2. **Single Layer Feedforward Network (SLFN)**: A supervised learning approach for classification tasks with optimized weight calculation

Both implementations are built from scratch without using high-level machine learning libraries, providing deep insights into the underlying mathematics and algorithms.

## ✨ Features

- **SOM Implementation**: Complete implementation of Kohonen's Self-Organizing Map algorithm
  - Unsupervised clustering
  - Topology-preserving mapping
  - Visualization of learned clusters

- **SLFN Implementation**: Single hidden layer feedforward neural network
  - Custom weight initialization
  - Forward propagation
  - Backpropagation from scratch
  - Multi-class classification capability

- **From-Scratch Development**: All core algorithms implemented without using scikit-learn or similar ML libraries
- **Comprehensive Visualizations**: Cluster maps, decision boundaries, and performance metrics
- **Detailed Documentation**: Well-commented code with mathematical explanations

## 🛠 Technologies Used

- **Python 3.x**
- **NumPy**: Numerical computations and matrix operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization and plotting
- **Seaborn**: Statistical data visualization
- **Jupyter Notebook**: Interactive development environment

## 📁 Project Structure
```
├── README.md
├── task1_som_clustering
│   ├── images
│   │   ├── som_hitmap_20x20_grid_final_train_data.png
│   │   ├── som_hitmap_4x4_grid_final_train_data.png
│   │   ├── som_samples_20x20_grid_final_train_data.png
│   │   └── som_samples_4x4_grid_final_train_data.png
│   └── som_clustering.ipynb
├── task2_som_classification
│   ├── images
│   │   ├── som_class_map_20x20_grid_test_set.png
│   │   ├── som_class_map_4x4_grid_test_set.png
│   │   ├── som_cm_20x20_grid_test_set.png
│   │   ├── som_cm_4x4_grid_test_set.png
│   │   ├── som_hitmap_20x20_grid_final_train_data.png
│   │   ├── som_hitmap_4x4_grid_final_train_data.png
│   │   ├── som_samples_20x20_grid_final_train_data.png
│   │   └── som_samples_4x4_grid_final_train_data.png
│   └── som_classification.ipynb
└── task3_slfn_titanic
    ├── images
    │   ├── slfn_confusion_matrix.png
    │   ├── slfn_loss_curve.png
    │   └── slfn_roc_curve.png
    ├── slfn_titanic.ipynb
    └── titanic.csv
```

## 🚀 Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AlieShayan/Neural-Networks-SOM-Clustering-and-SLFN-Classification.git
    cd Neural-Networks-SOM-Clustering-and-SLFN-Classification
    ```


2. **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```


3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```


4. **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

## 📊 Key Analyses & Visualizations

This project is structured into three main analysis tasks:

* **SOM Clustering (`task1_som_clustering`):**
    * **Objective:** Cluster the sklearn handwritten digits dataset using unsupervised learning.
    * **Technique:** Implements a Self-Organizing Map (SOM) to map high-dimensional data (64 features) onto a 2D grid.
    * **Visualizations:** Generates **Hit Maps** to show neuron activation frequencies and sample grids to visualize the digits captured by specific neurons.

* **SOM Classification (`task2_som_classification`):**
    * **Objective:** Utilize the topological properties of SOMs to perform classification on the digits dataset.
    * **Technique:** Labels neurons based on the majority class of the training data mapped to them, then predicts classes for test data.
    * **Visualizations:** Produces **Class Maps** and **Confusion Matrices** to evaluate classification performance across different grid sizes (e.g., 4x4 vs 20x20).

* **SLFN Classification (`task3_slfn_titanic`):**
    * **Objective:** Predict survival on the Titanic dataset.
    * **Technique:** Implements a Single Layer Feedforward Network (SLFN) to process passenger data.
    * **Visualizations:** Includes **Loss Curves** to track training progress and **ROC Curves** to analyze model sensitivity and specificity.


## 🔬 Methodology

### Self-Organizing Map Algorithm

1. **Initialize** weight vectors randomly for each neuron in the map
2. **Select** a random input vector from training data
3. **Find BMU** (Best Matching Unit) - the neuron with weights closest to input
4. **Update** weights of BMU and its neighbors based on distance
5. **Repeat** steps 2-4 for all iterations with decreasing learning rate

### SLFN Training Process

1. **Initialize** weights and biases randomly
2. **Forward propagation**: Compute activations through hidden and output layers
3. **Calculate loss** using appropriate loss function
4. **Backward propagation**: Compute gradients
5. **Update weights** using gradient descent
6. **Iterate** until convergence or max epochs reached

## 📊 Results

### SOM Clustering Performance
- Successfully clusters data into meaningful groups
- Preserves topological structure of input space
- Visualization shows clear separation between clusters

### SLFN Classification Metrics
- Training accuracy: [Add your results]%
- Testing accuracy: [Add your results]%
- Confusion matrix demonstrates strong classification performance across all classes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Alie Shayan**

- GitHub: [@AlieShayan](https://github.com/AlieShayan)
- Repository: [Neural-Networks-SOM-Clustering-and-SLFN-Classification](https://github.com/AlieShayan/Neural-Networks-SOM-Clustering-and-SLFN-Classification)

## 📚 References

1. Kohonen, T. (1990). The Self-Organizing Map. *Proceedings of the IEEE*
2. Huang, G.-B., et al. (2006). Extreme Learning Machine: Theory and Applications
3. Haykin, S. (2009). *Neural Networks and Learning Machines*
4. Bishop, C. M. (1995). *Neural Networks for Pattern Recognition*

---

⭐ **If you find this project useful, please consider giving it a star!**

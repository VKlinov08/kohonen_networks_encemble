# Kohonen Networks Ensemble for Image Classification

This project is my Master's thesis, where the use of an ensemble classifier for image classification based on a reference database was investigated. Each image in the reference database is memorized by a separate Kohonen network, which are trained in parallel. The classifier is designed to determine to which class from the reference database an input image belongs, based on the similarity of key point descriptors (ORB was chosen as the keypoint detector algorithm, using OpenCV implementation).

## Technologies and Libraries

- **NumPy** and **SciPy**: Used for core mathematical operations and algorithms.
- **scikit-learn**: Provides the `AccuracyScore` function for evaluating classification performance.
- **concurrent**: Used for parallelizing the training of the ensemble.
- **OpenCV**: Used for computing key point descriptors with the ORB detector.
- **Matplotlib**: Utilized for plotting graphs and displaying images.

## Installation and Setup

1. Clone the repository or download the project.
2. Open the project in **PyCharm** (or your preferred IDE).
3. Install the required dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   
4. Run `main.py` to start the demo:

   ```bash
   python main.py
   ```
   The demonstration function from the `demo.py` module will be executed. The model will train on the default image dataset and attempt to classify the images based on its learned "knowledge" (note that the fourth image may be misclassified, which is expected).


## Content

If you would like to compare the performance of the ensemble classifier with a standard Kohonen network (where one neuron is assigned to each class), you can run `experiment_1`.

To observe how the classification success rate depends on the number of neurons in the Kohonen network for each image, run `experiment_2`.

If you're interested in analyzing the learning and classification speed based on the number of neurons, run `experiment_3`.

The main classes are organized within the `models` module, while functions for image loading are located in the `utils` module.

To reduce the size of the keypoint descriptors and decrease computational load, the `descriptors_filtration.py` module is provided. You can observe the impact of the reduction by running the `show_reduction_impact` function in `main.py`.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

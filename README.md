# Explicit Quantum Surrogates for Quantum Kernel Models

This repository contains the **official implementation** of the paper *"Explicit Quantum Surrogates for Quantum Kernel
Models"* (A. Nakayama *et al.*, 2024 – [arXiv:2408.03000](https://arxiv.org/abs/2408.03000)).
It provides a **quantum-classical pipeline** to convert an implicit quantum machine learning model (a quantum kernel
SVM) into an **explicit quantum circuit model**, termed an *Explicit Quantum Surrogate (EQS)*.
The code is written in Python and uses **Qulacs** (for quantum state simulation) and **JAX** (for automatic
differentiation and optimization), allowing researchers to **reproduce the experiments and plots from the paper** and to
explore further extensions.

## Project Overview

Quantum machine learning models can be implicit (using quantum kernels in models like SVMs) or explicit (using
parameterized quantum circuits).
Implicit models often yield low training error but come with high prediction costs (and can be prone to overfitting),
while explicit models offer low prediction costs but can be difficult to train (e.g., due to barren plateaus).
This project implements the hybrid algorithm introduced in the paper to bridge these approaches:

- We first **train a quantum kernel SVM** on a quantum dataset (implicit model).
- We then **diagonalize an observable of the implicit models** derived from the trained SVM via eigenvalue
  decomposition.
- Using an **extended Automatic Quantum Circuit Encoding (AQCE)** algorithm, we **construct a quantum circuit** that
  explicitly encodes the top eigenvectors of that observable.
- Finally, we can fine-tune the EQS circuit as a Quantum Neural Network (QNN) on data to improve performance, combining
  fast quantum predictions with the flexibility for additional training when needed.

This approach combines the strengths of implicit and explicit QML: it reduces prediction costs (a single circuit
evaluation instead of costly kernel computations) and mitigates training difficulties by initializing the circuit from
the well-trained kernel model.

## Dependencies

The code is built and tested with Python 3 (Python 3.9+ recommended).
All required packages are listed in **`requirements.txt`**. You can install them with pip as shown below.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Akiraichi/Explicit-quantum-surrogates.git 
   cd Explicit-quantum-surrogates
   ```

2. **Install Python requirements**:  
   It is recommended to use a virtual environment (venv or conda). Then run:
   ```bash
   pip install -r requirements.txt
   ```  
   This will install all necessary libraries. (Note: Qulacs may require a working C++ build environment, but pip will
   usually provide a pre-built wheel.)

## Repository Structure and Usage

The core of the implementation is organized into a series of **numbered Python scripts (`01_*.py` through `05_*.py`)**
which should be run **in sequence** to reproduce the full pipeline from the paper.
Each stage generates intermediate results that feed into the next stage.
Below is a brief overview of each script and how to use it:

1. **`01_train_svm.py` – Train Quantum Kernel SVM (Implicit Model)**  
   *What it does:* Loads a dataset of quantum-encoded data and trains a Support Vector Machine (SVM) using a **quantum
   kernel**. The kernel is defined via the fidelity between quantum states (using Qulacs to simulate the states). This
   results in an implicit quantum model (the kernel SVM).  
   *Details:*
    - It uses scikit-learn’s `SVC` with a custom kernel function that computes fidelity between quantum state encodings
      of data points.
    - By default, it leverages a provided dataset (e.g., the **MNISQ dataset**, which encodes MNIST images into 10-qubit
      quantum states). You can modify the dataset or parameters via the `svm_util.select_svm_condition()` function (
      which defines the dataset name, number of qubits, number of training samples per class, SVM hyperparameter C,
      etc.).
    - The script splits data into training and test sets, trains a **One-vs-Rest** multi-class SVM (training one binary
      SVM per class), and prints the training and test accuracy.
    - **Output:** After training, it saves a Python object (using `joblib`) containing the trained SVM parameters and
      data (`alphas`, support vectors, intercepts, dataset splits, accuracies, etc.). This saved file (path defined in
      the SVM condition) will be used in subsequent steps.

   *Usage:* Run the script directly to train the SVM with default settings:
   ```bash
   python 01_train_svm.py
   ```  
   (Ensure you have the required dataset available; the code may download or generate it via the `mnisq` library if
   needed. Cached data will be reused on subsequent runs.)

2. **`02_eigenvalue_decompose.py` – Eigenvalue Decomposition of SVM Observable**  
   *What it does:* Performs eigenvalue and eigenvector decomposition on an **observable derived from the trained SVM
   model**. In essence, this script takes the SVM’s learned parameters and computes a Hermitian operator (measurement
   operator) that represents the decision function of the SVM in the quantum feature space, then finds its eigenvalues
   and eigenvectors. This step corresponds to **diagonalizing the implicit model** – a crucial step in constructing the
   explicit surrogate.  
   *Details:*
    - It loads the saved SVM results from step 1 (using the same SVM condition to locate the file).
    - Two methods are available: a newer approach (using `util_get_eigenvector_from_data`) and a legacy direct matrix
      diagonalization (controlled by an `old_experiment` flag). By default, the newer method is used, which avoids
      explicitly constructing large matrices and can incorporate simulated measurement noise if desired.
    - It computes the top eigenvalues and associated eigenvectors of the measurement operator. These eigenvectors (in
      statevector form) are the candidate basis states for the explicit quantum model.
    - Optionally, you can simulate **measurement noise** by specifying `noisy=True` and a number of shots (e.g., in the
      code, a call `process_task(noisy=True, n_shots=1000000)` is shown commented out for illustration). This will
      perturb the calculation as if the fidelity measurements were done with finite shots, to test robustness.
    - **Output:** Saves a data structure (via `joblib`) containing the eigenvalues and eigenvectors for each SVM
      classifier (for each class in One-vs-Rest). This will be used in the next step.

   *Usage:* Run the script after training the SVM:
   ```bash
   python 02_eigenvalue_decompose.py
   ```  
   By default it will perform a noiseless eigen-decomposition. To experiment with noise or the legacy method, you can
   edit the bottom of the script (uncomment the relevant lines or adjust flags).

3. **`03_convert_implicit2explicit.py` – Construct Explicit Quantum Surrogate (EQS) Circuit**  
   *What it does:* Uses the **Extended AQCE (Automatic Quantum Circuit Encoding)** algorithm to convert the implicit
   model into an **explicit quantum circuit**. This script takes the eigen-decomposition results from step 2 and finds a
   parameterized quantum circuit (a PQC) that has those eigenvectors encoded. Essentially, it **embeds the important
   eigenstates into a quantum circuit structure**, creating the initial Explicit Quantum Surrogate model.  
   *Details:*
    - It loads the SVM condition and eigen-decomposition results (from steps 1 and 2) for a given scenario. You can
      specify which SVM model (if multiple) and which set of eigenvectors to use via parameters `data_index` (index of
      the classifier/eigen-data to use) and `k` (how many top eigenvectors to embed). In the code, these are either
      passed to `process_task` or defined in a loop at the bottom. For example, `data_index=0` and `k=2` would take the
      first SVM (first class) and embed the top 2 eigenstates into the circuit.
    - It sets up an **AQCE condition** (via `select_aqce_condition`) to define the quantum circuit search parameters (
      number of qubits, etc., which should match the SVM’s number of qubits). It then uses the `ExtendedAQCE` class to
      find a circuit (or set of circuit components) whose state preparation aligns with the eigenvectors from the
      implicit model.
    - **Output:** The resulting quantum circuit (or its parameterization) is saved to disk. Specifically, the
      `ExtendedAQCE` object saves a list of unitary matrices (circuit building blocks) and their parameters to a file (
      the path is determined by the SVM and eigen decomposition config, and current date/time). This file represents the
      **EQS circuit** for the chosen class and eigenvector count.

   *Usage:* Before running, you may want to adjust the `data_index` and `k` values to select which model and how many
   eigenvectors to encode. By default (in the provided script bottom), it may run for a single case.
   ```bash
   python 03_convert_implicit2explicit.py
   ```  
   This will produce and save the explicit circuit for that case. To generate circuits for all classes in a multi-class
   problem, you can loop over `data_index` 0 to N-1 (N classes) and desired k values (as hinted in the commented example
   using `ProcessPoolExecutor` in the script). Ensure the `os.chdir(...)` path is set appropriately or removed (it was
   used in development; you can set it to the repository path if needed so that saved files go to a known location).

4. **`04_train_qnn.py` – Train the Quantum Surrogate (QNN) with the EQS Circuit**  
   *What it does:* Constructs a **Quantum Neural Network classifier** using the explicit circuit from step 3 and trains
   it (if desired) to improve performance. This script basically **integrates the EQS circuit into a trainable model**,
   evaluates its baseline performance (without additional training), then performs additional training (via gradient
   descent, using JAX/Optax) and evaluates the improved performance. This corresponds to fine-tuning the explicit model,
   and the results can be directly compared to the original SVM.  
   *Details:*
    - It first defines an experiment configuration via
      `get_defined_qnn_condition(exp_name, circuit_type, target_label)`. Here, `exp_name` might denote the dataset or
      experiment scenario (e.g., `"mnisq-mnist-001"` for a specific split of the MNISQ dataset), `circuit_type`
      specifies the type of circuit initialization (`"predefined"` to use the EQS from AQCE, or alternatives like
      `"random"` for baseline random circuit, etc.), and `target_label` picks which class label to focus on (for
      one-vs-rest training). By default in the script’s `__main__`, it loops over `target_label` 0 through 9 (for
      10-class classification) using `circuit_type="predefined"` – meaning it will use the circuits from step 3 as the
      starting point for each class.
    - It uses `prepare_experiment_conditions` to load the corresponding SVM, AQCE, and eigen decomposition
      configurations that match the `qnn_condition`. This ensures it loads the correct saved circuit (`matrixC_list`
      representing the circuit unitary) and eigenvalues (`lambda_k_list_list`) and SVM intercepts (`b_list`) for the
      model.
    - Next, it calls `create_eqs_circuit(qnn_condition, matrixC_list)` to build a **parametric circuit object** (
      `MyLearningCircuit`) that incorporates the fixed structure from the EQS. This circuit has trainable parameters (if
      any were left free by AQCE or for additional layers).
    - The original training dataset is loaded and split for “additional learning.”
    - It then evaluates **baseline performance** by using the EQS circuit’s predictions *without further training*, both
      on the data it was originally fitted on and on new test data. This gives a benchmark accuracy before fine-tuning (
      this corresponds to using the surrogate as derived directly from the kernel model).
    - The script then **trains the QNN** by calling `train_qnn(...)`, which runs a gradient-based optimization (using
      JAX for efficient gradient computation through the circuit and Optax for the optimizer) to minimize the
      classification loss on the training data. This adjusts the circuit’s parameters (if any) to improve accuracy. The
      training history (cost, gradients, parameters by iteration) is recorded.
    - After training, it evaluates the **new performance** (`evaluate_performance`) on training and test sets and prints
      the accuracies. This allows comparison of before-vs-after training.
    - **Output:** The script saves a dictionary (`save_data_dict` via `joblib`) containing the trained QNN parameters,
      final accuracies (train/test baseline and post-training), the conditions/configurations used, and the training
      history (e.g. parameter theta history, cost history over iterations). The saved file is named according to the QNN
      experiment name and settings.

   *Usage:* Before running, ensure you have executed steps 1–3 for all required classes/labels that you intend to
   train (so that the explicit circuits are available). Then run:
   ```bash
   python 04_train_qnn.py
   ```  
   With the default loop, it will train a separate QNN for each class label 0–9 using the predefined EQS circuits. You
   can adjust the parameters at the top of the script (`exp_name`, `circuit_type`, `target_labels`) to match your
   scenario.

5. **`05_predict.py` – Use the Trained Surrogate for Prediction**  
   *What it does:* Demonstrates how to load the trained explicit quantum surrogate model (from step 4) and use it to
   make predictions on data. This script essentially **combines the per-class QNNs into a full multi-class predictor**
   and evaluates its accuracy, verifying the end-to-end success of the surrogate. This corresponds to the inference
   stage, showing the reduced prediction cost in practice.  
   *Details:*
    - It defines a `main_predict(exp_name, circuit_type, target_labels, data_type, use_trained_parameter)` function.
      This function will loop through each `target_label` specified (each one corresponds to a one-vs-rest classifier)
      and load the trained QNN model for that label (using the `exp_name` and `circuit_type` to identify the saved file
      from step 4). It reconstructs the QNN (circuit and parameters) for each class.
    - For each class’s QNN, it retrieves the SVM intercepts and eigenvalues (though these may not all be needed just for
      prediction if the QNN is fully formed) and rebuilds the circuit via `create_eqs_circuit` (then applies the trained
      parameters if `use_trained_parameter=True`). It wraps it into a `MyQNNClassifier` object which can output decision
      values.
    - It then loads the original dataset (the same data used in training the SVM in step 1) via
      `svm_condition.load_dataset()`. Depending on `data_type`, it will either use the training portion or test portion
      of that original dataset for evaluation.
    - Each QNN (one per class) is used to produce a decision score (the SVM decision function value) for each sample.
      These are aggregated into a decision matrix of shape `[n_classes, n_samples]`. The script then takes `argmax` over
      classes for each sample to assign a predicted label. This mimics how a One-vs-Rest classifier makes a multi-class
      decision (choosing the class with highest confidence).
    - Finally, it compares the predicted labels to the true labels and computes the accuracy. This accuracy after
      additional training is printed out. In a multi-class scenario, this should match the test accuracy printed in step
      4 for the combined model.
    - This step essentially shows that the explicit surrogate (the QNNs) can be used on new data quickly: all the heavy
      lifting of creating the model is done, and prediction just involves evaluating the circuit outputs and a simple
      argmax. This is much faster than computing a kernel against all support vectors for each new sample, highlighting
      the **inference efficiency** of the EQS.

   *Usage:* Simply run after training is complete:
   ```bash
   python 05_predict.py
   ```  
   The default parameters in `__main__` will load the experiment `"mnisq-mnist-001"` results for all labels 0–9, use the
   `"random_structure"` circuit type (you may switch this to `"predefined"` to use the same circuits as trained), and
   evaluate on the test set (`data_type="test"`). It prints the accuracy of the surrogate on that test set. You can
   adjust `exp_name`, `circuit_type`, and `use_trained_parameter` (if you want to compare using the circuit *with* the
   trained parameters vs. just the initial parameters from AQCE). Typically, you would keep `use_trained_parameter=True`
   to see the full power of the trained surrogate.

## Citation

If you find this repository useful in your research or project, please consider citing the arXiv paper:

```bibtex
@article{nakayama2024explicit,
    title = {Explicit quantum surrogates for quantum kernel models},
    author = {Nakayama, Akimoto and Morisaki, Hayata and Mitarai, Kosuke and Ueda, Hiroshi and Fujii, Keisuke},
    journal = {arXiv preprint arXiv:2408.03000},
    year = {2024}
}
```
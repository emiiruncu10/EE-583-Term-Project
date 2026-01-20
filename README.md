# 🌍 Accuracy vs. Robustness: EuroSAT Classification with ConvNeXt

### **Project Definition**
This project evaluates the trade-off between **accuracy** and **robustness** in satellite image classification. While modern State-of-the-Art (SOTA) architectures like **ConvNeXt** achieve near-perfect accuracy on clean benchmarks, they often suffer from brittleness when exposed to real-world sensor noise.

This repository provides a comparative analysis of:
* **Classical Baselines:** K-Nearest Neighbors (K-NN), Random Forest (RF), Support Vector Machines (SVM).
* **Shallow Deep Learning:** A custom 3-layer CNN (`SimpleCNN`).
* **SOTA Models:** ConvNeXt (Tiny & Small variants) trained using both **Frozen** (Transfer Learning) and **Fine-Tuned** strategies.

**Key Finding:** We expose a "Glass Cannon" effect where fine-tuned models achieve ~99% accuracy on clean data but drop by ~50% under Gaussian noise ($\sigma=0.4$), whereas frozen models and classical baselines remain significantly more stable.

---

### **🚀 Quick Start (Google Colab)**

The easiest way to replicate the results is to use the provided Jupyter Notebook in Google Colab.

#### **Step 1: Open in Colab**
1.  Navigate to the file `EuroSAT_Model_Evaluation_Demo.ipynb` in this repository.
2.  Click the "Open in Colab" badge (if available) or download the file and upload it to [colab.research.google.com](https://colab.research.google.com/).

#### **Step 2: Runtime Configuration (GPU)**
This project requires a GPU for efficient execution.
1.  Go to **Runtime** > **Change runtime type**.
2.  Set **Hardware accelerator** to **T4 GPU**.
    * *Note:* If T4 is unavailable (or if you are on the Free tier), select **any available GPU**. The code is compatible with all standard Colab GPUs.
    * *Warning:* Running on CPU is possible but will be significantly slower for the Deep Learning sections.

#### **Step 3: Upload Trained Weights**
To skip the multi-hour training process and jump straight to the evaluation/demo, you must upload the pre-trained model weights provided in this repository (under the `models/` folder).

1.  In the Colab sidebar, click the **Folder icon** (Files).
2.  Upload the following 4 files:
    * `best_model_tiny_frozen.pth`
    * `best_model_tiny_finetuned.pth`
    * `best_model_small_frozen.pth`
    * `best_model_small_finetuned.pth`
    * *(Optional: `best_model_simple_cnn.pth` - if not uploaded, the notebook will train this lightweight model from scratch in ~1 minute).*

#### **Step 4: Run the Notebook**
1.  Click **Runtime** > **Run all**.
2.  The notebook will automatically:
    * Download and extract the **EuroSAT dataset**.
    * Train/Evaluate the **Classical Baselines** (RF, SVM, KNN).
    * Train the **SimpleCNN** (if weights are missing).
    * Load your uploaded **ConvNeXt** weights.
    * Perform the **Noise Robustness Test** and generate comparison charts.

---

### **📂 Repository Structure**

```text
├── EuroSAT_Model_Evaluation_Demo.ipynb  # Main Verification Suite (Run this)
├── ConvNeXt.ipynb                       # (Reference) Original training script
├── models/                              # Pre-trained weights (Upload to Colab)
│   ├── best_model_tiny_frozen.pth
│   ├── best_model_tiny_finetuned.pth
│   ├── best_model_small_frozen.pth
│   └── best_model_small_finetuned.pth
├── reports/                             # Final PDF Report & Presentation
└── README.md                            # You are here

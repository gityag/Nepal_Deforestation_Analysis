# Geospatial Deep Learning for Deforestation Mapping in Nepal

### A Comparative Analysis of CNN Architectures for Semantic Segmentation of Satellite Imagery

---

## Project Overview

This project is an end-to-end deep learning pipeline for semantic segmentation of satellite imagery to identify forest cover and deforestation in Nepal. It leverages Google Earth Engine for data acquisition and PyTorch for model development. The core of the project is a rigorous, comparative analysis of four distinct experimental setups to solve the key challenges of the task, namely severe class imbalance and training instability in complex architectures.

The primary goal is to leverage the historical accuracy of the Hansen Global Forest Change dataset to train a model that can make predictions on recent, high-resolution Sentinel-2 imagery, creating a powerful workflow for near-real-time environmental monitoring.

- **Data Sources:** Sentinel-2 (Input Images) & Hansen Global Forest Change (Labels)
- **Primary Frameworks:** PyTorch, Google Earth Engine API
- **Models Compared:**
  1.  **Baseline FCN:** A simple Fully Convolutional Network with unweighted loss.
  2.  **Weighted FCN:** The FCN trained with a weighted loss to counter class imbalance.
  3.  **Unstable U-Net:** A U-Net trained with weighted loss, demonstrating instability with a high learning rate.
  4.  **Stable U-Net:** The U-Net trained with weighted loss and a learning rate scheduler, our winning model.

---

## Key Results: Model Comparison

The final analysis shows that the **Stable U-Net** is the most effective model. It successfully incorporates the weighted loss to learn the rare "Deforested" class while its superior architecture and stabilized training produce the best balance of performance across all land cover classes.

| Metric | Baseline FCN | Weighted FCN | Unstable U-Net | **Stable U-Net (Winner)** |
|:---|:---:|:---:|:---:|:---:|
| **Mean IoU (mIoU)** | **0.498** | 0.436 | 0.314 | 0.465 |
| **Macro F1-Score**| **0.569** | 0.532 | 0.419 | 0.551 |
| **Recall (Deforested)** | 0.000 | 0.580 | **0.817** | 0.417 |
| **Precision (Deforested)**| 0.000 | **0.011** | 0.006 | 0.008 |

---

## Visual Analysis of Model Progression

The side-by-side comparison below perfectly illustrates the project's story. The Baseline FCN ignores deforestation (red). The Weighted FCN learns to see it, but its predictions are noisy. The Stable U-Net produces the cleanest, most spatially coherent maps, demonstrating its superior architecture.

![Prediction Example](./side-by-side-comparison.png)

---

## How to Run

The entire workflow is contained within the `Nepal_deforestation_analysis.ipynb` Jupyter Notebook. The notebook is designed to be run sequentially in an environment like Google Colab.

### Requirements:
1.  A Google account with **Google Earth Engine access**.
2.  A Google Colab environment with a **`GCP_PROJECT_ID` secret** containing a valid GEE-linked project ID.
3.  Required Python libraries are installed at the beginning of the notebook (`earthengine-api`, `geemap`, `rasterio`).

### Workflow Steps:
1.  **Phase 1: Foundational Analysis:** Uses the Google Earth Engine API to analyze the Hansen dataset and establish a baseline of forest change in Nepal.
2.  **Phase 2: Data Preparation & Model Definition:** Contains cells to export training patches from GEE (a time-intensive, one-time setup), define the PyTorch `Dataset`, and define all model architectures and master functions.
3.  **Phase 3: Model Training:** Four modular cells train each experimental model sequentially. Each training run is fully checkpointed and can be resumed.
4.  **Phase 4: Final Evaluation:** The final section loads all saved models and generates the dynamic comparison table and the final side-by-side visualizations.

---

## Conclusion & Future Work

### Conclusion
This project successfully demonstrates the effectiveness of using a U-Net architecture with a weighted loss function for mapping deforestation. A key remaining challenge is the low **precision** on the "Deforested" class, where the winning model still confuses naturally bright surfaces like riverbeds with cleared land.

### Future Work
To further improve performance, future experiments could include:
-   **Advanced Architectures (e.g., Vision Transformer):** A Transformer-based architecture like **SegFormer** could be implemented to leverage global context and better distinguish between features like rivers and deforestation patches.
-   **Data Augmentation:** Applying random geometric and color augmentations to the training data to improve model robustness.
-   **Post-Processing:** Using techniques like Conditional Random Fields (CRFs) to clean up and refine the final prediction maps.
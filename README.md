# MIRACLE: A Continual Integration Method for Single-cell Data

<div align="center">
  <img src="docs/source/_static/img/Figure1.png" alt="MIRACLE Logo" width="900px">
</div>

<p align="center">
By employing <strong> dynamic architecture adaptation and data rehearsalstrategies </strong>, MIRACLE enables <strong> continual integration  </strong> of diverse datasets while preserving biological fidelity over time.
</p>

<p align="center">
  <a href="https://github.com/sc-miracle/miracle/stargazers"><img src="https://img.shields.io/github/stars/sc-miracle/miracle?style=social" alt="GitHub Stars"></a>
  <!-- <a href="https://pypi.org/project/scmidas/"><img src="https://img.shields.io/pypi/v/scmidas" alt="PyPI version"></a> -->
  <a href="https://scmiracle.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/scmiracle" alt="Documentation Status"></a>
  <a href="https://github.com/sc-miracle/miracle/LICENSE"><img src="https://img.shields.io/github/license/sc-miracle/miracle?v=1" alt="License"></a>
</p>

---

**MIRACLE** , an online integration framework for multimodal single-cell integration via continual learning (CL). CL enables models to incrementally incorporate new data while preserving previously acquired knowledge. MIRACLE employs CL strategies including dynamic architecture adaptation and data rehearsal to enhance adaptability and knowledge retention. It leverages MIDAS as a base model to support the integration of multimodal mosaic data, enabling online updates across diverse omics. MIRACLE also 
facilitates precise query mapping to reference atlases, improving the accuracy of cell label transfer and novel cell discovery.

- **MIRACLE Documentation:** [**scmiracle.readthedocs.io**](https://scmiracle.readthedocs.io/en/latest/)
- **MIDAS Documentation:** [**scmidas.readthedocs.io**](https://scmidas.readthedocs.io/en/latest)
<!-- - **Publication:** [***Nature Biotechnology***](https://www.nature.com/articles/s41587-023-02040-y) -->

## ✨ Key Features
For MIRACLE:
*    **Boosted Efficiency with Continual Integration**: Incrementally add new data batches to an existing model, which eliminates the need for complete retraining and significantly reduces computational requirements.

For base model (MIDAS):
*    **Multi-Modal Support**: Natively supports RNA, ADT, and ATAC data, and can be easily configured to incorporate additional modalities.
*   **Mosaic Data Integration**: Seamlessly integrates datasets where different batches measure different sets of modalities (e.g., some samples have RNA and ATAC, while others have only RNA).

*   **Data Imputation**: Accurately imputes missing modalities, turning incomplete data into a complete multi-modal matrix.
*   **Batch Correction**: Effectively removes technical variations between different batches, enabling consistent and reliable analysis across datasets.
*   **Efficient and Scalable**: Built on PyTorch Lightning for highly efficient model training, with support for advanced strategies like Distributed Data Parallel (DDP).
*   **Advanced Visualization**: Integrates with TensorBoard for real-time monitoring of training loss and UMAP visualizations.

## 🚀 Installation

Get started with MIDAS by setting up a conda environment.

```bash
git clone https://github.com/sc-miracle/miracle.git
conda create -n scmiracle python=3.12
conda activate scmiracle
cd miracle
pip install -r docs/source/requirements.txt
pip install -e .
```

## ⚡ Getting Started

To get started, please refer to our [documentation](https://scmiracle.readthedocs.io/en/latest/).

## 📈 Reproducibility

To reproduce the results from our publication, please visit the `reproducibility` branch of this repository:
[**https://github.com/sc-miracle/miracle-reproducibility/**](https://github.com/sc-miracle/miracle-reproducibility)

<!-- ## 📜 Citation

If you use MIRACLE in your research, please cite our paper:

He, Z., Hu, S., Chen, Y. *et al*. Mosaic integration and knowledge transfer of single-cell multimodal data with MIDAS. *Nat Biotechnol* (2024). https://doi.org/10.1038/s41587-023-02040-y

```bibtex
@article{he2024mosaic,
  title={Mosaic integration and knowledge transfer of single-cell multimodal data with MIDAS},
  author={He, Zhen and Hu, Shuofeng and Chen, Yaowen and An, Sijing and Zhou, Jiahao and Liu, Runyan and Shi, Junfeng and Wang, Jing and Dong, Guohua and Shi, Jinhui and others},
  journal={Nature Biotechnology},
  pages={1--12},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
``` -->

<!-- ## 🙌 Contributing

We welcome contributions from the community! If you have a suggestion, bug report, or want to contribute to the code, please feel free to open an issue or submit a pull request. -->

## 📝 License

 MIRACLE is available under the [MIT License](https://github.com/sc-miracle/miracle/LICENSE).

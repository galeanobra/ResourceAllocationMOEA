# Optimizing load-balanced resource allocation in next-generation mobile networks: A parallelized multi-objective approach

This repository contains the data and scripts associated with the study *Optimizing Load-Balanced resource allocation in next-generation mobile networks: A parallelized multi-objective approach*, published in the journal *Ad Hoc Networks (https://doi.org/10.1016/j.adhoc.2025.103912). The study proposes a novel parallelized multi-objective evolutionary approach to optimize resource allocation in next-generation mobile networks.

## Repository Structure

The repository is organized as follows:

- **results/**: Contains the results obtained from the optimization process.
  - **raw/**: Stores the raw optimization results, including:
    - **FUN files**: Objective function values for each solution in the final population.
    - **VAR files**: Decision variable values corresponding to each solution.
    - The data is further separated into subfolders based on topology type (`low`, `hybrid` and `high`).
  - **att/**: Contains attainment surfaces, as described in Section 4.1 of the paper.
- **kriging/**: Contains a script for training a surrogate model based on Kriging. This surrogate model was used to analyze the feasibility of replacing expensive evaluations with a more computationally efficient approach.
- **plots.ipynb**: A Jupyter Notebook with scripts to generate visualizations from the results.

## Data Description

Each topology folder in *raw/* contains 30 FUN and 30 VAR files in CSV format, corresponding to 30 different instances of the topology.

- **FUN files**: Each row represents a solution from the final population of the multi-objective evolutionary algorithm (MOEA), with values corresponding to different objectives.
- **VAR files**: Each row represents the decision variables of the corresponding solution in the FUN file.

## Notes

- The study is currently under review and has not been published yet.
- The dataset and scripts are provided to support reproducibility and further research.
- If you have any questions or need additional details, feel free to contact us.

## Citation

If you use this repository in your work, please cite the original article (this section will be updated once the paper is published):

```
@article{calle2025optimizing,
  title={Optimizing load-balanced resource allocation in next-generation mobile networks: A parallelized multi-objective approach},
  author={Calle-Cancho, Jes{\'u}s and Galeano-Brajones, Jes{\'u}s and Cort{\'e}s-Polo, David and Carmona-Murillo, Javier and Luna-Valero, Francisco},
  journal={Ad Hoc Networks},
  volume={177},
  pages={103912},
  year={2025},
  doi={10.1016/j.adhoc.2025.103912},
  publisher={Elsevier}
}
```

## License

The contents of this repository, including code, data, and results, are provided solely for academic and research purposes. Use of the materials requires proper citation of the original article. Any commercial use, redistribution, or modification without explicit permission from the authors is strictly prohibited.

---

For any questions or further clarifications, please contact the authors.







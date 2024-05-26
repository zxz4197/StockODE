# Stockode Project README

## Data Preprocessing (preprocess)

In the `preprocess` folder, you will find code related to data preprocessing.

All data are under the [data folder](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/tree/master/data).(from Feng et al.)

## Running the Project

Ensure that your system supports GPU and follow these steps to run the Stockode project(Refer to the Dependencies):

   ```bash
   python main.py --gpu
   ```
## Dependencies
- Python version==3.7.0
- Torch version==1.4.0
- Libraries:
   ```bash
   pip install -r requirement.txt
   ```
## Note*
- We implemented a simple method such as ARIMA by ourselves. We replicate these baseline methods by utilizing links to the open-source code given in the corresponding papers or by requesting the source code from them, such as RSR. For newly added baselines, we strictly follow their papers to conduct the experiments.However, because we do not have access to the latest source code of the three baselines, we can only ensure that each module of the papers are strictly implemented and the experimental settings are consistent with the other baselines, but there may be still some details that are inconsistent with the original papers.


- If you encounter any issues, refer to the accompanying documentation or contact me!

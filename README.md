# NRAO_ALMA_MSDS_Capstone_2024

## Usage
You can run our code with the following steps:
* Fork this repo
* Navigate to the root repo directory in a terminal window
* Save your project title and abstract separated by ". " in a file named "input.txt"
* Run `python app.py`

From this point the script will print the following information for the project with the title and abstract in "input.txt"
* The predicted probability of the project having only continuum measurements (being a "continuum" project) as predicted by a logistic regression using the TF-IDF vectorized text
* The predicted probability of the project having at least one spectral line measurement (being a "line" project) as predicted by a logistic regression using the TF-IDF vectorized text
* The predicted topic number out of 50 topics generated by Latent Dirichlet Allocation
* The top two predicted bands from Multinomial Naive Bayes
* The locally hosted address of a dashboard initialized to the predicted topic and bands above
  * The dashboard will be hosted at http://127.0.0.1:8050/ once initialized
The goal of this project is to assist those writing proposals to use the Atacama Large Millimeter/Submillimeter Array (ALMA) telescope. Specifically, we aim to encourage more precise and effective observations with suggestions of frequency parameters based on the research proposal text. 

### Additional dashboard resources
A version of the dashboard not tied to user inputs is here: https://capstonedashboard-8.onrender.com/

Notice that you can filter for your specified "topic" and "bands." Then, you can pay attention to the high scoring clusters for this specific proposal text. The hope is that through our model's predictions, you can look at the distribution of frequency ranges used by previous, similar projects. We suggest you take the count of observations in each frequency range and the width of each frequency range into account.

## Dependencies
The following libraries are required for this code
* pandas
* numpy
* nltk
* re
* scikit learn
* string
* joblib
* plotly
* plotly express
* plotly dash
* [ALminer](https://alminer.readthedocs.io/en/latest/) (necessary for updating data but not for "app.py")

## File Organization
### Dashboard Folder
The "Dashboard" folder holds .py and jupyter notebook of dashboard code that does not require inputs. You can use these files to explore each topic without the predictions from models.

### Data Folder
* **Model Outputs**
  * Holds primary outputs of models trained on full data (as opposed to train/test split models used for accuracy measures, discussed later)
  * These are used to power the "app.py" code
* **Raw Data**
  * "nrao_projects.csv" contains all 4,528 projects cleaned and processed from "Data_Ingestion.ipynb"
  * "observations.csv" contains the output from [https://almascience.nrao.edu/aq/](https://almascience.nrao.edu/aq/) used to find "line" and "continuum" measurements as ALminer does not include this data. See "Data_Ingestion.ipynb" for further details on data ingestion and cleaning.
  * "train_measurements.zip" and "test_measurements.zip" include the measurement-level data for all spectral line projects, split into training and testing groups. These must be extracted to .csv files if you wish to use them.
  * "train_projects.csv" and "test_projects.csv" contain the project-level data for spectral line projects, split into training and testing groups.
* **Notebooks**
  * Various notebooks used for ingesting and exploring data
  * "Data_Ingestion.ipynb" is by far the most important here as it creates the datasets described above. If you have any questions about the data, refer to this notebook. 

**Models Folder**
The "Models" folder holds the individual models used to power this project.
* The ".joblib" files hold the models trained on the full dataset and are imported and used to power "app.py"
  * These are equivalent to "pickling" models, just in sklearn's preferred format
* The "train_test.ipynb" notebooks hold the train/test split code used to obtain accuracy measurements and are valuable for understanding model performance and re-training.
* The "full_data.ipynb" notebooks use the training and testing data **combined** to create the production models found in the ".joblib" files.

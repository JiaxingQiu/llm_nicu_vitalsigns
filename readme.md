## Project: Can LLMs describe NICU HR and SPO2 events?
- Prepare data.
    + download data from UVA dataverse: https://dataverse.lib.virginia.edu/dataset.xhtml?persistentId=doi:10.18130/V3/5UYB4U
    + unzip and save as folder "data"
- scripts are under "script" folder. 
    + main.R and utils are for data pre-processing
    + CLIP: CLIP model for HR and SPO2 time serise VS. events description.
    + VAE: VAE model for HR and SPO2 time serise encoders.
    + WSDIST: within sample distance for negative sampleing for "LLM describing time series" project empirical evaluation.
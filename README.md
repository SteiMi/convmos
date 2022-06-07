# ConvMOS: Climate Model Output Statistics with Deep Learning

This repository contains code for the paper "ConvMOS: Climate Model Output Statistics with Deep Learning".

## Required data
* REMO data
    * This data is available [here](https://professor-x.de/datasets/remo_convmos.tgz) (~ 5.6 GB).
    * It is already prepared for use with ConvMOS.
* GTOPO 30
    * GTOPO is provided by [USGS](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-global-30-arc-second-elevation-gtopo30?qt-science_center_objects=0#).
    * Make sure you remap it to the same grid as the REMO data with `cdo` for example.
* E-OBS
    * Available from [ECAD](https://www.ecad.eu/download/ensembles/download.php). We used V19.0e for our paper.
    * Make sure you remap it to the same grid as the REMO data with `cdo` for example.

Correct paths to the data have to be set in the `config.ini`.

## Data preparation (if you have raw REMO data)
This code is designed to work with data from REMO RCM. To this end, we have REMO variables as netcdf-files at the EUR-44 grid and GER-11 grid.  We base this on data with timesteps of 6-hours, but we want daily data. These files have to be preprocessed for them to be used here. This is achieved with `prepare_netcdf_files.sh`.

First, I made sure that the data is available in the following structure with .nc files in the leave nodes:
```
EUR-44 or GER-11
    {var1withdifferentpressures}
        {pressure1}
        {pressure2}
    {var2withoutdifferentpressures}
    {var3withdifferentpressures}
        {pressure1}
        {pressure2}
```

Sometimes, there are accidental `a.nc` files in these folders which have to be removed.

Then, I ran the script for each var-folder like, for example, so `./prepare_netcdf_files.sh EUR-44/var1withdifferentpressures -d -f` **EXCEPT** for variable `260` (precipitation), which I preprocess with `./prepare_netcdf_files.sh EUR-44/260 -d -f -s`, so that it calculates the daily sum and not the mean of the 6-hour parts.

While it most probably does nothing bad to leave it like that, I also commented out the `remapbic` cdo call and changed the output of the daily aggregation step to `${filename}_remap.nc` for GER-11, as the original data provided is already in GER-11 grid, so no remapping is necessary. Note that I left it in once and compared the resulting files with `ncdiff` and the differences were all 0, so I'm pretty sure that this is not necessary, but I did it anyways.

I also used this script on the E-OBS target data. Make sure the data is on the same grid and that the variables are renamed, so that xarray can read the files.

## Running the models
* Baselines:
    * Rename `config_baseline.ini` to `config.ini`
    * Adjust `model` to the baseline model you want to run (see `baseline.py` for available model names)
    * Run `python3 baseline.py`
* ConvMOS:
    * Adjust `config.ini` to your liking
    * Run `python3 run.py`

## Running the models in Kubernetes
We use the following models:

* Lin (Local Linear Regression):
    * Adjust `config_template_remo_prec_baseline.ini` so that `aux_variables` is empty (if necessary)
    * Run `start_k8s_job.sh my_lin_run -m linear`
* NL PCR (Non-local Principal Component Regression):
    * Adjust `config_template_remo_prec_baseline.ini` so that `aux_variables` contains all variables (if necessary)
    * Run `start_k8s_job.sh my_nlpcr_run -m linear-nonlocal`
* NL RF (Non-local Random Forest):
    * Adjust `config_template_remo_prec_baseline.ini` so that `aux_variables` contains all variables (if necessary)
    * Run `start_k8s_job.sh my_nlrf_run -m rf-nonlocal`
* ConvMOS:
    * Adjust `config_template_remo_prec.ini` so that `aux_variables` contains all variables (if necessary)
    * Run `start_k8s_job.sh my_convmos_run -m convmos -a gggl -s`
        * `gggl` specifies the module composition

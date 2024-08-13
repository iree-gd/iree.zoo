# IREE Zoo

This repository builds models from **kaggle** into IREE models. You can either use models already built here, or fork the repo and build yourself (after enabling actions on main branch).

## Download Model job

Downloads a single model using a kaggle url. Works only for tflite.

## Multiple Models job

Downloads all models with version specified in job. This is going to be good for rebuilding models when a new version of iree appears.

## Models

Current models available are in the Releases of this repo.

## Run locally

In order to convert tflite model, run:

```bash
python3 convert_tflite.py kaggle/esrgan-tf2/tfLite/esrgan-tf2
```

This will create the following folder structure:

```bash
build/addons/iree-zoo/kaggle_esrgan_tf2_tfLite_esrgan_tf2/
```

Inside the folder with the name of url (but / and - replaced with _) you will find the **iree models** and a **gdscript** file to run it.

# Introduction
A simple streamlit application to capture and convert image using laptop camera.

## Launch
As per usual with streamlit

```sh
streamlit run app.py
```

Or debug mode
```sh
streamlit run app.py --logger.level=debug
```


## Setup
Make sure you have a .env file with the following keys

```
HUGGINGFACEHUB_API_TOKEN=<YOUR_HUGGINGFACEHUB_API_TOKEN>
CUDA_VISIBLE_DEVICES=""
```
Also run the following in your favourite virtual environment.

```
pip install -r requirements.txt 
```

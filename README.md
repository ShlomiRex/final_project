# Final Project

Title: Text-Conditioned Image Generation with Stable Diffusion

## Virtual Environment Setup (important!)

We have to make sure we are using the same python version and packages as my development environment in order not to break my project.

Install `virtualenv`, `pyenv`, `poetry`. Then run:

```
pyenv local 3.11.8
poetry env use 3.11.8
poetry install
poetry env activate
```

Make sure that the last command `poetry env activate` run again in your terminal to make sure the active environment is running (example):

`source /Users/user/Library/Caches/pypoetry/virtualenvs/text-conditioned-image-generation-using-st-l7k_OWS4-py3.11/bin/activate` (or on Windows (for example): `C:\Users\Shlomi\AppData\Local\pypoetry\Cache\virtualenvs\text-conditioned-image-generation-using-st-35DVCAXA-py3.11\Scripts\activate.ps1`)

We can be sure we created a new environment by running `pip list`, we should see only pip in the list. If not, we can remove the environment folder (given by `poetry env info`).

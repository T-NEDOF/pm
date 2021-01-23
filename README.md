#### FLASK RESTFUL API

### Terminal commands

    Initial installation: make install

    To run test: make tests

    To run application: make run

    To run all commands at once : make all


### Viewing the app ###

    Open the following url on your browser to view swagger documentation
    http://127.0.0.1:5000/


### Using Postman ####

    Authorization header is in the following format:

    Key: Authorization
    Value: "token_generated_during_login"

    For testing authorization, url for getting all user requires an admin token while url for getting a single
    user by public_id requires just a regular authentication.

### Full description and guide ###
https://medium.freecodecamp.org/structuring-a-flask-restplus-web-service-for-production-builds-c2ec676de563


### Contributing
If you want to contribute to this flask restplus boilerplate, clone the repository and just start making pull requests.

```
https://github.com/cosmic-byte/flask-restplus-boilerplate.git

```

#### PYTHON PACKAGE MODELS

### Setup model package ###
    Go to package folder, run command:
    python regression_model/setup.py sdist

### Install model package ####
    Go to package folder, run command:
    pip install dist/regression_model-1.0.0.tar.gz

### Using model package ####
    Just import the package into your module, then you can use model pipeline to predict:
    from regression_model.predict import make_prediction
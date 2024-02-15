############### Build image ##################
FROM ubuntu:18.04
FROM python:3.11 AS build

# using this label we can clean up current cached images
LABEL delete_when_outdate=yes

    # python
ENV PYTHONUNBUFFERED=1
    # prevents python creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONHASHSEED=random

    # pip
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100
#    \
    # pipenv variables maybe ??
#    PIPENV_VERSION=1.2.2 \
#    PIPENV_VIRTUALENVS_CREATE=false \
#    PIPENV_CACHE_DIR='/var/cache/pypipenv'

# Install pipenv and compilation dependencies
# TODO: Make "code" directory via docker-compose file
WORKDIR /code/backend/
# TODO: Зачем два раза копировать? Сначала Pipfile, потом code/backend
COPY Pipfile Pipfile.lock ./
RUN python -m pip install --upgrade pip
RUN pip install pipenv && pipenv install
# Install python dependencies in /.venv
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy --system --ignore-pipfile
RUN pip install torch torchvision

# Code
COPY . /code/backend/

RUN mkdir -p /code/backend/www/public

RUN chmod -R 777 /code/backend/www/public

RUN chmod +x /code/backend/docker/entrypoint.sh


CMD ["echo", "backend image"]

FROM tlamadon/balke-lamadon-2022-aer

ENV HOME=/tmp

# create user with a home directory
USER root

ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

RUN cp -r /app/ $HOME

WORKDIR ${HOME}
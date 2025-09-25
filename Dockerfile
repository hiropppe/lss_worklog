FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config locales tzdata \
    python3.10-dev neovim python3-neovim git curl jq less \
    ca-certificates \
    mysql-client libmysqlclient-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen ja_JP.UTF-8

ENV LANG=ja_JP.UTF-8
ENV TZ=Asia/Tokyo
ENV PATH="${PATH}:/root/.local/bin"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN echo 'eval "$(uv generate-shell-completion bash)"' >> $HOME/.bashrc

# R
RUN apt-get update -qq
RUN apt-get install -y --no-install-recommends software-properties-common dirmngr gfortran cmake liblapack-dev libblas-dev \
    libxml2-dev libssl-dev libcurl4-openssl-dev libfontconfig1-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
    libharfbuzz-dev libfribidi-dev pkg-config libudunits2-dev libgdal-dev
RUN apt-get install -y wget
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
RUN apt-get install -y --no-install-recommends r-base
RUN apt-get install -y --no-install-recommends gpg-agent
RUN add-apt-repository ppa:c2d4u.team/c2d4u4.0+
RUN apt-get install -y --no-install-recommends r-cran-rstan r-cran-devtools r-cran-ggmcmc r-cran-gtools r-cran-irkernel
# Install R packages step by step
# Configure R to avoid compilation errors with Rcpp packages
RUN echo 'CXXFLAGS += -Wno-format-security' >> /usr/lib/R/etc/Makeconf
RUN echo 'CPPFLAGS += -Wno-format-security' >> /usr/lib/R/etc/Makeconf

RUN R -e "install.packages('magrittr', repos='https://cran.rstudio.com/')" && echo "magrittr installed"
RUN R -e "install.packages(c('systemfonts', 'textshaping'), repos='https://cran.rstudio.com/')" && echo "systemfonts/textshaping installed"
RUN R -e "install.packages('ragg', repos='https://cran.rstudio.com/')" && echo "ragg installed"
RUN R -e "install.packages(c('selectr', 'rvest'), repos='https://cran.rstudio.com/')" && echo "selectr/rvest installed"
RUN R -e "install.packages('tidyverse', repos='https://cran.rstudio.com/')" && echo "tidyverse installed"
RUN R -e "install.packages('minqa', repos='https://cran.rstudio.com/')" && echo "minqa installed"
RUN R -e "install.packages('RcppEigen', repos='https://cran.rstudio.com/')" && echo "RcppEigen installed"
RUN R -e "install.packages('pbkrtest', repos='https://cran.rstudio.com/')" && echo "pbkrtest installed"
RUN R -e "install.packages('lme4', repos='https://cran.rstudio.com/')" && echo "lme4 installed"
RUN R -e "install.packages('car', repos='https://cran.rstudio.com/')" && echo "car installed"
RUN R -e "install.packages('rstatix', repos='https://cran.rstudio.com/')" && echo "rstatix installed"
RUN R -e "install.packages('ggpubr', repos='https://cran.rstudio.com/')" && echo "ggpubr installed"

# Verify packages are installed
RUN R -e "print(installed.packages()[,'Package'])" && echo "Package verification completed"
RUN R -e "install.packages('quanteda')"
RUN R -e "install.packages('quanteda.textmodels')"
RUN R -e "install.packages('quanteda.textstats')"
RUN R -e "install.packages('quanteda.textplots')"

# エラー: Failed to install 'quanteda.corpora' from GitHub:
#   名前空間 ‘processx’ 3.5.2 はロードされていますが、>= 3.6.1 が要求されています
RUN R -e "install.packages('processx')"
RUN R -e "devtools::install_github('quanteda/quanteda.corpora')"

RUN R -e "install.packages('LSX')"

# required for rpy2
RUN apt-get install -y --no-install-recommends libpcre2-dev libbz2-dev liblzma-dev libicu-dev libblas-dev libdeflate-dev

RUN apt-get install -y fonts-ipafont fonts-ipaexfont
RUN fc-cache -fv

RUN apt-get install -y graphviz

# Add Quarto
RUN apt-get install -y --no-install-recommends npm nodejs \
    && npm install -g n \
    && n stable \
    && apt purge -y npm nodejs

RUN git clone -b v1.8.21 --depth 1 https://github.com/quarto-dev/quarto-cli.git \
    && cd quarto-cli \
    && ./configure.sh

ARG USERNAME=take
ARG USER_UID=1000
ARG USER_GID=1000

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
 && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN mkdir -p /workspaces/lss
COPY . /workspaces/lss
RUN chown -R $USERNAME:$USERNAME /workspaces

# Fix R library permissions for the user
RUN chmod -R 755 /usr/local/lib/R/site-library

# Final verification before completion
RUN ls -la /usr/local/lib/R/site-library/ && echo "Library directory listing completed"
RUN R -e "pkgs <- installed.packages()[,'Package']; print(paste('ggpubr:', 'ggpubr' %in% pkgs)); print(paste('minqa:', 'minqa' %in% pkgs)); print(paste('tidyverse:', 'tidyverse' %in% pkgs))" && echo "Final package verification completed"

WORKDIR /workspaces/lss

RUN git config --global --add safe.directory /workspaces/lss

RUN uv sync --reinstall

CMD ["uv", "run", "jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--ServerApp.token=''", "--no-browser"]

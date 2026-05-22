FROM danteev/texlive

# Git LFS: required because the monorepo tracks *.pt via .gitattributes at repo root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# Install tools
RUN pip install arxiv-latex-cleaner==0.1.* pdflinkchecker-cli==0.2.* --break-system-packages
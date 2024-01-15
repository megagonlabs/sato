# Adapted from https://github.com/varlen/sato-env

# TODO: Run these commands once, then disable
conda create --name sato-docker python=3.9
conda activate sato-docker

# Define the base path for your application
export BASEPATH="/home/tazik/sato"
export SHERLOCKPATH="$BASEPATH/sherlock"
export EXTRACTPATH="$BASEPATH/extract"
export PYTHONPATH="$PYTHONPATH:$SHERLOCKPATH:$BASEPATH"
export TYPENAME="type78"

# Install dependencies
conda install numpy=1.21.4 pandas=1.3.1 flask scikit-learn=0.24.2
pip install torch==1.9.0 gensim==3.8.3 nltk==3.6.2
pip install ConfigArgParse tensorboardX

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download glove 6B
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# To run your server, use: cd demo; python server.py

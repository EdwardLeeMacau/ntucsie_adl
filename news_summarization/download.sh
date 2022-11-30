# Download pre-trained model for multiple choice
if [ ! -d ./ckpt/google-mt5-small ]; then
  mkdir -p ./ckpt
  wget https://www.dropbox.com/s/jbd6x9c6s038z3f/google-mt5-small.tar.gz?dl=0 -O ./google-mt5-small.tar.gz
  tar zxvf ./google-mt5-small.tar.gz --directory ./ckpt
  mv ./ckpt/google-mt5-small.backup ./ckpt/google-mt5-small
fi

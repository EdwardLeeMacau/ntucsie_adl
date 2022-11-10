# Download pre-trained model for multiple choice
if [ ! -d ./ckpt/multiple_choice ]; then
  mkdir -p ./ckpt/multiple_choice
fi

if [ ! -d ./ckpt/multiple_choice/bert-base-chinese.best ]; then
  wget https://www.dropbox.com/s/w6ve5g72fymlohf/bert-base-chinese.tar.gz?dl=1 -O ./bert-base-chinese.tar.gz
  tar zxvf ./bert-base-chinese.tar.gz --directory ./ckpt/multiple_choice
fi

# Download pre-trained model for question answering
if [ ! -d ./ckpt/question_answering ]; then
  mkdir -p ./ckpt/question_answering
fi

if [ ! -d ./ckpt/question_answering/hfl-chinese-roberta-wwm-ext.best ]; then
  wget https://www.dropbox.com/s/qqcz848nnr3ogb0/hfl-chinese-roberta-wwm-ext.tar.gz?dl=1 -O ./hfl-chinese-roberta-wwm-ext.tar.gz
  tar zxvf ./hfl-chinese-roberta-wwm-ext.tar.gz --directory ./ckpt/question_answering
fi

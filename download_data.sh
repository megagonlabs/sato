cd $BASEPATH

data_URL="http://sato-data.s3.amazonaws.com"

curl "$data_URL/tmp.zip" > tmp.zip
unzip tmp.zip

curl "$data_URL/pretrained.zip" > sherlock/pretrained.zip
cd sherlock; unzip pretrained.zip; cd -

curl "$data_URL/LDA_cache.zip" > topic_model/LDA_cache.zip
cd topic_model; unzip LDA_cache.zip; cd -

# clean up
rm tmp.zip
rm sherlock/pretrained.zip
rm topic_model/LDA_cache.zip

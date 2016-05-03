if [ -z output ]; then
    mkdir output
fi
python sst_utils.py
cat phraseTrain.txt sentencesTrain.txt > output/train.txt
cat sentencesTest.txt > output/test.txt

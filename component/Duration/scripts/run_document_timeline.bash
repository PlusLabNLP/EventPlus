#!/bin/bash  
#base_dir=$(pwd)
base_dir=$(cd ../ && pwd)
cd ../stanford-corenlp-full-2018-10-05

#docname="sample_document.txt"
java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP [ -props sampleProps.properties ] -file ../input_data/$1 -outputFormat conllu
mv *.output $base_dir/input_data_conllu/

cd ../scripts
python run_model.py -doc ../input_data_conllu/$1.output -gpu 0 -out ../predictions


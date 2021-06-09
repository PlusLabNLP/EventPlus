#### Steps to create a document timeline for an input document ###

1. Put all the input document files into the "input_data" folder. Note that each document file should have sentences separated by a "\n". A "sample_document.txt" file is already present as a reference for an input document file.

2. From the terminal, change the current directory to be the "scripts" folder and run the following command:
	bash run_input_data.bash

3. The predictions of all the input document files will be written to the predictions folder:
    - [input_doc_filename]_timeline.csv  (contains the document timeline)
    - [input_doc_filename]_predictions.csv (contains the relative timelines and predicate durations)


The mappings for durations are as follows:
0-inst
1-secs
2-mins
3-hrs
4-days
5-weeks
6-mnths
7-yrs
8-decs
9-cents
10-forever


For a detailed description of the protocols, datasets, as well as models of these data, please see the following paper:
Vashishtha, S., B. Van Durme, & A.S. White. 2019. Fine-Grained Temporal Relation Extraction. arXiv:1902.01390 [cs.CL].  (https://arxiv.org/abs/1902.01390)
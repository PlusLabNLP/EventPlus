#!/bin/bash
base_dir=$(cd ../ && pwd)

#!/bin/bash
for filename in $(ls $base_dir/input_data/); do
    bash run_document_timeline.bash "$filename" 
done
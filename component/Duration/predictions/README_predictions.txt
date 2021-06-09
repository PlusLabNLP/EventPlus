## Data dictionary for sample_document.txt.output_predictions.csv:

1. Each row corresponds to an event-pair in a sentence.

2. sent_pred_id1: filename sent_id pred_position
	For eg (row 1): sample_document.txt.output 1 4 denotes predicate at the 4th position (index starting at 0) in the 1st sentence of 'sample_document.txt.output' file.

3. sent_pred_id2: same as above
	For eg (row 1): sample_document.txt.output 1 8 denotes predicate at the 8th position (index starting at 0) in the 1st sentence of 'sample_document.txt.output' file.

Note that there are two sentence ids because the full sentence is the concatenation of the sentence in the sent_pred_id1 and the next adjacent sentence in the document. 

Examples:
The 1st row in the csv file has:
sent_pred_id1: sample_document.txt.output 1 4
sent_pred_id2: sample_document.txt.output 1 8

which denotes that both the predicates in the predicate-pair are being considered from the 1st sentence and are at 4th and 8th position.

The 4th row in the csv file has:
sent_pred_id1: sample_document.txt.output 1 13
sent_pred_id2: sample_document.txt.output 2 4

which denotes that the first predicate is at the 13th position in the 1st sentence and the second predicate is at the 4th position in the 2nd sentence in the document.

4. B1: beginning point of the first predicate

5. E1: end point of the first predicate

6. B2: beginning point of the second predicate

7. E2: end point of the second predicate






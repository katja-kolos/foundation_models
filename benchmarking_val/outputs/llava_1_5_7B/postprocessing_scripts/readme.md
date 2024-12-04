# ReadMe

`benchmarking.ipynb` is the script that I (Katja) used to obtain the metrics in the metrics folder. It did 3 things: 
* parse raw output into answer and solution
* map answer string to answer index
* perform evaluation (accuracy, textual similarity) -- with my version of Dasha's version of the scripts of ScienceQA repository

We later decided that each model's results should be stored not only raw, but also parsed (yes, we could have thought of it right away, but if we were smart we would not be doing this at all). 
So we now store `parsed_json` with solution (renamed to `explanation` for some reason) + answer number, and also the postprocessing scripts. 

So, the notebook `Parse Outputs.ipynb` in this folder repeats part of `benchmarking.ipynb` necessary to obtain the parsed `csv`s. 

Also we are not consistent in how we name the folders with raw outputs: `json_output`, `json_outputs`, `json_format` (angry comment ommitted). 

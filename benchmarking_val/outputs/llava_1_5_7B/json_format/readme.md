# ReadMe

Note: This output was generated in 3 runs because of memory limits of the MacBook it was run on: 
- run 1: 
	- rows 0 - 1614 of filtered_validation_data, QTCH
	- rows 0 - 1614 rows of filtered_validation_data, QTCHL
	- rows 0 - 1613 rows of filtered_validation_data, QTCHLS
	- rows 0 - 1613 rows of filtered_validation_data, QTCHS
- run 2:
	- rows 1615 - 3214 of filtered_validation_data, QTCH
	- rows 1615 - 3214 of filtered_validation_data, QTCHL
	- rows 1614 - 3214 of filtered_validation_data, QTCHLS
	- rows 1614 - 1911 of filtered_validation_data, QTCHS
- run 3:
	- rows 1912 - 3214 of filtered_validation_data, QTCHS
- run 4:
	- row 3215 of filtered_validation_data, all settings

The model used is LLaVA1.5-7b from huggingface: `llava-hf/llava-1.5-7b-hf`

The issue with empty responses was fixed: empty responses resulted from the inability of the model to work with None images. A blank image was passed as placeholder with 'Ignore image' suffix. 

Still there:
Issue 2: The output format currently in the answer field is not the number of the correct answer but the correct answer as a string. I still need to check if this has a match with one of the choices in 100% cases. 

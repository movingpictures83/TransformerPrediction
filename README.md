# TransformerPrediction
# Language: Python
# Input: TXT
# Output: PNG
# Tested with: PluMA 1.1, Python 3.6

PluMA plugin that uses a transformer to predict future points.

The plugin expects as input a tab-delimited file of keyword value pairs:
datapath: Dataset
steps: Number of steps
modelconfig: JSON file
metainformation: JSON file
predictionname: Output directory

PNG images will be sent to the output directory

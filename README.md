#######################
## set up
#######################
### you should first set up the system.

First install pip:

`sudo apt-get install python-pip python-dev`

Then install tensorflow:

`export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl
sudo pip install --ignore-installed --upgrade $TF_BINARY_URL`

Then install pandas:

`sudo apt-get install python-pandas`


#########################
## usage
#########################

run the ‘letterRecCNN.py’ file, you can use ‘—-help’ option to show the usage.

```
Usage: python letterRecCNN.py [options]

Options:
  -h, --help           show this help message and exit
  --train              Train the neural network, should also specify data with
                       --datafile=FILE
  --test               Test the neural network with input data --datafile=FILE
  --crossValid         10-fold corss validation with input data
                       --datafile=FILE
  --datafile=DATAFILE  set input data file
```

### EXAMPLE:
if you want to test the neural network with data ‘DATAFILE.data’, you should use the following syntax:

`$ python letterRecCNN.py --test --datafile='DATAFILE.data'`

The predicted labels will be written into file ‘predicted_labels.txt’. The accuracy will be shown on the screen. 


if you want to train it with data ‘DATAFILE.data’, you should use the following syntax:
(Note that your training will overwrite the model I saved in folder ‘/modelCNN’. If it is overwritten, please copy the model files in ‘/modelCNN/backup’ to ‘/modelCNN’ for testing)

`$ python letterRecCNN.py —-train --datafile='DATAFILE.data'`



if you want to conduct 10-fold cross validation with data ‘DATAFILE.data’, you should use the following syntax:
`$ python letterRecCNN.py —-crossValid --datafile='DATAFILE.data'`


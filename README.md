# Image-Completion
In this project, we will remove a part of an image and try to reconstruct it. The images are resized to 256x256 and grayscaled to have 1 channel. The outpout size will also be the 256x256x1 but only the removed part will be considered as an output and for calculating the loss. The model used is a simple Autoencoder.

## Example Input
![alt text](https://github.com/kareemjano/Image-Completion/blob/main/Snapshots/in.png)

## The Model
In what follows I will be using an Autoencoder model. The model will take as an input the 256x256x1 with the 40x40 pixels in the middle set to 0. The output will also be 256x256x1 but the loss will only be computed on the blacked out pixels. It uses four convlution and pooling layer for the encoder and 4 upsampling and convlution layers for the decoder. Also Xavier/2 weights initialization was used which helps much in having faster convergence. More possible models will be discussed at the end of the notebook.
## Optimization and Visualization
Adam optimizer with MSE loss are used. The Trainer class will manage the training of the model and it supports an early stopping criteria, a schedular, TensorBoard logging(loss, accuracy and images), and saves a checkpoint of the model at the lowest loss value. Retraing the model is also possible (either from the checkpoint or by saving the moddle with the 'inference' flag as False).

The following is a small test for the model.

## Stopping Criteria
The patience flag passed to the Trainer along with the val_epoch controlls the stopping criteria. The number of bad epochs is calculated as following: if the validation epoch loss is larger than the minimumn validation loss a penalty (+1) is counted, and if it is greater an award is counted (-1) as long as the value remains above 0. If the number of bad_epochs = patience then the training will stop. The reason why the validation loss is tracked is because we want our model to generalize on unseen data.

## Checkpoint
When checkpoint_dir is assigned to the Trainer, the model with the minimum validation loss will be saved in the given directory.

## StepLR Schedular
The schedular will multiply the learning rate by 'gamma' every 'step_size' steps


## Hyperparameter Tuning
A small manual hypermarameter tunning was done. Nevertheless, this can be imporved by using some framework such as Optuna and many other.


Note: Run "tensorboard --logdir=runs" to open tensorboard

Note: A model checkpoint is attached. Therefore running the training again is not required. The model checkpoint can be loaded and then used for inference or for further training. Also the model's Tensorboard file is attached.

## Evaluation: 
### Accuracy
The accuracy is calculated based on 0.005 precision. That is the average loss of each sample is compared with 0 with a tolerance of 0.005 and if it is within the range a point will be rewarded. Then this value is summed for all samples and averages accross batches and then accross the whole training dataset.
### Loss
Since it is a regretion problem the MSELoss is also used as a metric to evaluate the model

## Results
### Tensorboard
![alt text](https://github.com/kareemjano/Image-Completion/blob/main/Snapshots/loss.png)
![alt text](https://github.com/kareemjano/Image-Completion/blob/main/Snapshots/acc.png)
![alt text](https://github.com/kareemjano/Image-Completion/blob/main/Snapshots/tb.png)

### On validation data
![alt text](https://github.com/kareemjano/Image-Completion/blob/main/Snapshots/eval_results.png)

### On unseen data
![alt text](https://github.com/kareemjano/Image-Completion/blob/main/Snapshots/results_unseen.png)
 
## Comments
### Choice of the model:
When talking about generative models, one can immediately think about Autoencoders, Variational Autoencoders (VAEs), or GANs. The difference between Autoencoders and VAEs or GANs is that the later ones predicts parameters for the distribution of the output and then an output is sampled from the distribution. Whereas, Autoencoders directy prident the output which suits our task better since we dont need to learn a distribution but we only car about one ouput. 
### Improving the model:
Even that the current model gives verry good results, some improvments can be made such as using a UNet model which also an autoencoder but links every level of decoder and encoder together through directly. This always longer training and prevents from overfitting.

Another proposal would be to use a pretrained decoder, fine tune it, and then train the encoder. This will allow the encoder to predict better image features rather than learning from images which are distorted in the middle.

### Generaliztion
As is shown in the last cell, the results were not as good as we expects. Collect more images or applying data augmentation could be a way overcome this problem and achieve better generalization.

### Snapshots
The runned notebook as a pdf, and snapshots from Tensorboard are attached to the snapshots folder.

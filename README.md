# AC-SUM-GAN: Connecting Actor-Critic and Generative Adversarial Networks for Unsupervised Video Summarization

## PyTorch Implementation of AC-SUM-GAN
- From **"AC-SUM-GAN: Connecting Actor-Critic and Generative Adversarial Networks for Unsupervised Video Summarization"** (IEEE Transactions on Circuits and Systems for Video Technology (TCSVT 2020))
- Written by Evlampios Apostolidis, Eleni Adamantidou, Alexandros I. Metsai, Vasileios Mezaris and Ioannis Patras
- This software can be used for training a deep learning architecture that embeds an Actor-Critic model into a Generative Adversarial Network for automatic video summarization. Training is performed in a fully unsupervised manner without the need for ground-truth data (such as human-generated video summaries). After being unsupervisingly trained on a collection of videos, the AC-SUM-GAN model is capable of producing representative summaries for unseen videos, according to a user-specified time-budget about the summary duration.

## Main dependencies
- Python  3.6
- PyTorch 1.0.1

## Data
Structured h5 files with the video features and annotations of the SumMe and TVSum datasets are available within the "data" folder. The GoogleNet features of the video frames were extracted by [Ke Zhang](https://github.com/kezhang-cs) and [Wei-Lun Chao](https://github.com/pujols) and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). These files have the following structure:
<pre>
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /gtscore                  1D-array with shape (n_steps), stores ground truth improtance score (used for training, e.g. regression loss)
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    /n_frames                 number of frames in original video
    /picks                    positions of subsampled frames in original video
    /n_steps                  number of subsampled frames
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
    /video_name (optional)    original video name, only available for SumMe dataset
</pre>
Original videos and annotations for each dataset are also available in the authors' project webpages:
- TVSum dataset: https://github.com/yalesong/tvsum
- SumMe dataset: https://gyglim.github.io/me/vsum/index.html#benchmark

## Training
To train the model using one of the aforementioned datasets and for a number of randomly created splits of the dataset (where in each split 80% of the data is used for training and 20% for testing) use the corresponding JSON file that is included in the "data/splits" directory. This file contains the 5 randomly generated splits that were utilized in our experiments.

For training the model using a single split, run:
<pre>
python main.py --split_index N (with N being the index of the split)
</pre>
Alternatively, to train the model for all 5 splits, use the 'run_splits.sh' script according to the following:
<pre>
chmod +x run_splits.sh    # Makes the script executable.
./run_splits              # Runs the script.  
</pre>
Please note that after each training epoch the algorithm performs an evaluation step, using the trained model to compute the importance scores for the frames of each video of the test set. These scores are then used by the provided evaluation scripts to assess the overal performance of the model (in F-Score).

The progress of the training can be monitored via the TensorBoard platform and by:
- opening a command line (cmd) and running: tensorboard --logdir=/path/to/log-directory --host=localhost
- opening a browser and pasting the returned URL from cmd

## Configurations
Setup for the training process:

- On 'data_loader.py', specify the path to the 'h5' file of the dataset and the path to the 'json' file containing data about the created splits.
- On 'configs.py', define the directory where the models will be saved to.
    
Arguments in 'configs.py': 
<pre>
--video_type: The used dataset for training the model. Can be either 'TVSum' or 'SumMe'.
--input_size: The size of the input feature vectors (1024 for GoogLeNet features).
--hidden_size: The hidden size of the LSTM units.
--num_layers: The number of layers of each LSTM network.
--regularization_factor: The value of the regularization factor (ranges from 0.0 to 1.0).
--entropy_coef: The entropy regularization coefficient delta (0.1 in this implementation).
--n_epochs: Number of training epochs.
--clip: The gradient clipping parameter.
--lr: Learning rate.
--discriminator_lr: Discriminator's learning rate.
--split_index: The index of the current split.
--action_state_size: The size of the action-state space (60 in this implementation).
</pre>
For the parameters with no explicitly defined default values, please read the paper ("Implementation Details" section) or check the 'configs.py' file.

## Model Selection and Evaluation
In order to perform the evaluation described in the paper (Section V.A, V.B), you must have trained the model using 10 different values of the regularization factor (by running the file run_splits.sh). Then you can run the 'pipeline.sh' file, after specifying the path to the folder where the results of the experiment are stored. 

## Citation
If you find this code useful in your work, please cite the following publication:

E. Apostolidis, E. Adamantidou, A. I. Metsai, V. Mezaris and I. Patras, **"AC-SUM-GAN: Connecting Actor-Critic and Generative Adversarial Networks for Unsupervised Video Summarization,"** in IEEE Transactions on Circuits and Systems for Video Technology.

DOI: 10.1109/TCSVT.2020.3037883

## License
Copyright (c) 2020, Evlampios Apostolidis, Eleni Adamantidou, Alexandros I. Metsai, Vasileios Mezaris, Ioannis Patras / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Acknowledgement
This work was supported by the European Union Horizon 2020 research and innovation programme under contract H2020-780656 ReTV. The work of Ioannis Patras has been supported by EPSRC under grant No. EP/R026424/1.

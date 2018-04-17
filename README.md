# Character_recognitionAI
Character recognition using Convolutional Neural Networtk

# Setting up
-> virtualenv -p python3 envcharrec

-> Activate virtualenv

-> pip3 install -r requirements.txt

# Train and Predict
-> python3 gendata.py

-> python3 train.py

-> python3 check_model.py

# Additional info
The trained model is always overwritten to best_weights.hdf5, for every tuning of the model any improvements to accuracy and loss will be saved in a seperate folder in the root directory

Example: 
    
    65per - 65% accurate weights

    75per - 75% accurate weights

Provided the architecture of the Neural network, these weights can be loaded and used for prediction.
This is demonstrated in check_model.py

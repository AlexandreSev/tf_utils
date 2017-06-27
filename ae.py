# coding: utf-8

import tensorflow as tf
import tf_utils

class ff_auto_encoder:
    """
    Simple feed forward auto-encoder
    """
    
    def __init__(self, input_shape=163968, hidden_sizes = [1024], optimizer="ADAM", learning_rate=0.001, 
                 denoising=False, stacked=True, partial=False):
        """
        Args:
            input_shape (int): Size of the input of the auto-encoder
            hidden_sizes ([int]): list of the size of the hidden layers of the encoder
            optimizer (string): Which optimizer to use. For now, only "ADAM" impleted.
            learning_rate (float): Learning rate of the optimizer.
            denoising (Boolean): If True, dropout is added to the input.
            stacked (Boolean): If True and if the autoencoder has several hidden layers, the first layer
                               will be trained train alone first, then the first and the second, and so on.
            partial (Boolean): Set to True for partial autoencoders coming from stacking.
        """
        
        self.denoising = denoising
        if denoising:
            self.dropout = tf.placeholder(dtype=tf.float32)
        else:
            self.dropout = 1.

        self.sizes = [input_shape] + hidden_sizes
        
        self.stacked = stacked & (len(hidden_sizes) > 1)
        if self.stacked:
            self.partial_aes = [ff_auto_encoder(input_shape, hidden_sizes[0: i], optimizer, 
                                learning_rate, denoising, False, True) for i in range(1, len(hidden_sizes)+1)]
        else:
            self.partial_aes = []

        self.partial = partial

        if optimizer == "ADAM":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            #TODO: Implement other optimizers
            raise ValueError("NOT IMPLEMENTED YET")
            
        self.build()
    
    def build(self):
        """
        Lead the construction of the tensorflow graph by calling appropriate functions.
        """
        self.create_weights()
        self.create_placeholder()
        self.build_model()
        
    def create_weights(self):
        """
        Create the weights of the model
        """
        self.params = {"weights": {}, "biases": {}}
        for i in range(len(self.sizes)-1):
            self.params["biases"]["encoder_b%s"%i] = tf_utils.create_bias_variable(shape=[self.sizes[i+1]])
            self.params["weights"]["encoder_W%s"%i] = tf_utils.create_weight_variable(shape=(self.sizes[i], self.sizes[i+1]))
            self.params["biases"]["decoder_b%s"%i] = tf_utils.create_bias_variable(shape=[self.sizes[i]])
            self.params["weights"]["decoder_W%s"%i] = tf_utils.create_weight_variable(shape=(self.sizes[i+1], self.sizes[i]))
            
    def create_placeholder(self):
        """
        Create the placeholder of the model.
        """
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, self.sizes[0]))
        
    def build_model(self):
        """
        Create the operation between weights and placeholders
        """
        current_input = tf.nn.dropout(self.input_tensor, self.dropout)
        
        for i in range(len(self.sizes)-1):
            current_input = tf.nn.relu(tf.matmul(current_input, self.params["weights"]["encoder_W%s"%i]) + \
                                      self.params["biases"]["encoder_b%s"%i])
            
        self.representation = current_input
        
        for i in range(len(self.sizes)-2, -1, -1):
            current_input = tf.nn.relu(tf.matmul(current_input, self.params["weights"]["decoder_W%s"%i]) + \
                                      self.params["biases"]["decoder_b%s"%i])
        
        self.out_tensor = current_input

        self.loss = tf.reduce_mean(tf.reduce_sum((self.input_tensor - self.out_tensor)**2, 1))

        if self.partial:
            n_temp = len(self.sizes) - 2
            list_var = [self.params["weights"]["encoder_W%s"%n_temp], self.params["weights"]["decoder_W%s"%n_temp], 
                        self.params["biases"]["encoder_b%s"%n_temp], self.params["biases"]["decoder_b%s"%n_temp]]
        else:
            list_var = None

        self.training_step = self.optimizer.minimize(self.loss, var_list=list_var)

    def training(self, sess, X_train, X_val=None, X_test=None, n_epoch=100, callback=True, saving=True, 
                 save_path="./model.ckpt", warmstart=False, weights_path="./model.kpt", display_step=50,
                 denoising_rate=0.75):
        """
        Train the model on given data.

        Args:
            sess (tensorflow.Session): Current running session.
            X_train (np.array): Training data. Must have the shape (?, input_shape)
            X_val (np.array): Data which determine the best model. Must have the shape (?, input_shape)
            X_test (np.array): Data on which is computed a score.
            n_epoch (int): Number of passes through the data
            callback (boolean): If true, all errors will be saved in a dictionary
            saving (boolean): If true, the best model will be saved in the save_dir folder.
            save_path (string): where to save the file if saving==True.
            warmstart (boolean): If true, the model will load weights from weights_path at the beginning.
            weights_path (string): Where to find the previous weights if warmstart=True.
            display_step (int): The number of epochs between two displays.
            denoising_rate (float): Keeping probability in the first dropout.

        Return:
            dictionnary: If callback==True, return the callback. Else, return an empty dictionnary
        """

        tf.set_random_seed(42)
        sess.run(tf.global_variables_initializer())

        if self.denoising:
            feed_dict = {self.input_tensor: X_train, self.dropout: denoising_rate}
        else:
            feed_dict = {self.input_tensor: X_train}

        if callback:
            dico_callback = {"training_error": [], "validation_error": [], "testing_error": []}
        else:
            dico_callback = {}

        if warmstart:
            tf_utils.loader(self.params, sess, weights_path)
        
        if X_val is not None:
            best_val_error = tf_utils.compute_reconstruction_error(self, X_val, sess)
            best_n_epoch = 0
        else:
            best_val_error = None
            best_n_epoch = None

        loading=False
        for i, ae in enumerate(self.partial_aes):
            loading=True
            warmstart_temp = warmstart | (i > 0)
            weights_path_temp = weights_path
            if (warmstart_temp & (i>0)):
                weights_path_temp = save_path
            print("Training partial autoencoder number %s"%i)
            ae.training(sess, X_train, X_val, X_test, n_epoch, callback, saving, save_path, 
                        warmstart_temp, weights_path_temp, display_step, denoising_rate)

        if loading:
            tf_utils.loader(self.params, sess, save_path)
        

        
        #Loop over epochs
        for nb_epoch in range(1, n_epoch + 1):
            
            # Run the training step
            dico_callback, best_val_error, best_n_epoch = self.training_one_step(nb_epoch, sess, 
                feed_dict, X_train, X_val, best_val_error, best_n_epoch, X_test, callback, 
                dico_callback, display_step, saving, save_path)

        if callback:
            return dico_callback
        else:
            return {}

    def training_one_step(self, nb_epoch, sess, feed_dict, X_train, X_val=None, best_val_error=None, 
                      best_n_epoch=None, X_test=None, callback=True, dico_callback={}, display_step=50, 
                      saving=True, save_path="./model.ckpt"):
        """
        Run the training for 1 epoch.

        Args:
            nb_epoch (int): Number of the current epoch
            sess (tensorflow.Session): Current running session.
            X_train (np.array): Training data. Must have the shape (?, input_shape)
            X_val (np.array): Data which determine the best model. Must have the shape (?, input_shape)
            best_val_error (float): Minimum error encountered yet.
            best_n_epoch (int): On which epoch best_val_error has been found.
            X_test (np.array): Data on which is computed a score.
            callback (boolean): If true, all errors will be saved in a dictionary
            dico_callback (dictionary): dictionary where callbacks are saved, if callback is True.
            display_step (int): The number of epochs between two displays.
            saving (boolean): If true, the best model will be saved in the save_dir folder.
            save_path (string): where to save the file if saving==True.
            
        Returns:
            dictionary: dictionary where callbacks are saved, if callback is True.
            float: Minimum error encountered yet.
            int: On which epoch the mininimum error has been found.
        """

        sess.run(self.training_step, feed_dict=feed_dict)
            
        if X_val is not None:
            val_error = tf_utils.compute_reconstruction_error(self, X_val, sess)
            if val_error < best_val_error:
                if saving:
                    tf_utils.saver(self.params, sess, save_path)
                best_val_error = val_error
                best_n_epoch = nb_epoch


        if callback:
            dico_callback["training_error"].append(tf_utils.compute_reconstruction_error(self, X_train, 
                                                                                         sess))
            
            if X_val is not None:
                dico_callback["validation_error"].append(val_error)
            
            if X_test is not None:
                dico_callback["testing_error"].append(tf_utils.compute_reconstruction_error(self, X_test, 
                                                                                            sess))
        
        if nb_epoch%display_step==0:
            self.display(nb_epoch, sess, X_train, X_val, X_test, best_val_error, best_n_epoch)

        return (dico_callback, best_val_error, best_n_epoch)

    def display(self, nb_epoch, sess, X_train, X_val=None, X_test=None, best_val_error=None, 
                best_n_epoch=None):
        """
        Display some usefull information

        Args:
            nb_epoch (int): number of current epoch.
            sess (tensorflow.Session): current running session.
            X_train (np.array): Training data.
            X_val (np.array): Data which determine the best model.
            X_test (np.array): Data on which is computed a score.
            best_val_error (float): Minimum error encountered yet.
            best_n_epoch (int): On which epoch best_val_error has been found.
        """
        print("Epoch %s"%nb_epoch)
        print("Train Error %s"%tf_utils.compute_reconstruction_error(self, X_train, sess))
        
        if X_val is not None:
            print("Validation Error %s"%tf_utils.compute_reconstruction_error(self, X_val, 
                                                                              sess))
            print("Best model: %s epochs with Error %s"%(best_n_epoch, best_val_error))
        
        if X_test is not None:
            print("Testing Error %s"%tf_utils.compute_reconstruction_error(self, X_test, sess))
        
        print("")
    
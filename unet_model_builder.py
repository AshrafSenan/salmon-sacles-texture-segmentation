import tensorflow as tf
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

## Default hyper-parameter settings

ACTIVATION = 'relu'
DROP_OUT_SIZE = 0.1
INITIALIZER = 'he_normal'
PADDING = 'same'
POOLING_SIZE = (2,2)
IS_FINAL_CONTR = False
STRIDES = (2, 2)
KERNEL_SIZE = (3,3)
L = 0.0001

class u_net_model:

    def __init__(self, input_size, num_of_classes, 
                 activation= ACTIVATION, 
                 drop_out_size = DROP_OUT_SIZE,
                 initializer = INITIALIZER,
                 padding = PADDING):
        
        ## Initialize the model hyper parameters
        self.input_size = input_size
        self.activation =activation
        self.drop_out_size = drop_out_size
        self.initialzier =  initializer
        self.padding = padding
        self.num_of_classes = num_of_classes
        self.Model = None
        
    def create_contraction_layer(self,prev_contr_layer,
                                 filters, 
                                 kernel_size, 
                                 strides = STRIDES,
                                 drop_out_size = DROP_OUT_SIZE,
                                 activation = ACTIVATION, 
                                 kernel_initializer= INITIALIZER, 
                                 padding= PADDING,
                                 pooling_size = POOLING_SIZE):
        """[Creates a U-Net contraction layer consist of two convolutions, one dropout and one pooling

        Args:
            prev_contr_layer (Tensofrlow Layer): The previous layer
            filters (int): [description]
            kernel_size (array): Kernel Size
            strides (array, optional): Strides. (2,2).
            drop_out_size (float, optional): . the percentatge of droupped out connections Defaults to 0.1
            activation (string, optional): The activation function. Defaults to 'relu'.
            kernel_initializer (string, optional): The kernel intitializer. Defaults to 'he_normal''.
            padding (string, optional): convolution padding. Defaults to 'same'.
            pooling_size (array, optional): Pooling window size. Defaults to POOLING_SIZE.

        
        Returns:
            CNN U-Net contraction Layer
        """    
        
        
        ## Make the first convloutional
        new_contr_layer = tf.keras.layers.Conv2D( filters, 
                                                  kernel_size,            
                                                 activation = activation, 
                                                 kernel_initializer=kernel_initializer, 
                                                 padding = padding) (prev_contr_layer)
        #Drop out some connections to avoid overfitting
        new_contr_with_drop = tf.keras.layers.Dropout(drop_out_size)(new_contr_layer)
        
        ## Make the second convolution
        new_contr_layer2 = tf.keras.layers.Conv2D( filters,  kernel_size,                                                   
                                                  activation = activation, 
                                                 kernel_initializer=kernel_initializer, 
                                                 padding = padding)(new_contr_with_drop)
       
        new_contr_layer_pooled = tf.keras.layers.MaxPooling2D(pooling_size)(new_contr_layer2)
        
        
        return new_contr_layer_pooled, new_contr_layer2
      
        
    def create_expansive_layer(self,contr_layer, prev_layer, 
                                filters, 
                                kernel_size, 
                                strides, 
                                drop_out_size = DROP_OUT_SIZE,
                                activation = ACTIVATION, 
                                kernel_initializer= INITIALIZER, 
                                padding= PADDING,
                                pooling_size = POOLING_SIZE):

        """Creates a U-Net expansion layer consist of transpose, two convolutions, one dropout 

        Args:
            prev_contr_layer (Tensofrlow Layer): The corresponding previous layer
            prev_layer(Tensorflow Layer): The previous expansion Layer
            filters (int): [description]
            kernel_size (array): Kernel Size
            strides (array, optional): Strides. (2,2).
            drop_out_size (float, optional): . the percentatge of droupped out connections Defaults to 0.1
            activation (string, optional): The activation function. Defaults to 'relu'.
            kernel_initializer (string, optional): The kernel intitializer. Defaults to 'he_normal''.
            padding (string, optional): convolution padding. Defaults to 'same'.
            pooling_size (array, optional): Pooling window size. Defaults to POOLING_SIZE.

        
        Returns:
            CNN U-Net expansion Layer
        """       
        transposed_layer = tf.keras.layers.Conv2DTranspose( filters, 
                                                           (2,2), 
                                                           strides = strides,                                                            
                                                           kernel_initializer = kernel_initializer,
                                                           padding = padding)(prev_layer)
        conc_layer = tf.keras.layers.concatenate([transposed_layer, contr_layer])
        
        conv1_layer = tf.keras.layers.Conv2D(filters, 
                                            kernel_size, 
                                            activation= activation,
                                            kernel_initializer = kernel_initializer,
                                            padding = padding)(conc_layer)
        
        dropped_layer = tf.keras.layers.Dropout(drop_out_size)(conv1_layer)
        
        conv2_layer = tf.keras.layers.Conv2D(filters, 
                                            kernel_size, 
                                            activation=activation,
                                            kernel_initializer= kernel_initializer,
                                            padding = padding)(dropped_layer)
        
        return conv2_layer
    
    def create_contraction_layer_regularized(self,prev_contr_layer,
                                 filters, 
                                 kernel_size, 
                                 strides = STRIDES,
                                 drop_out_size = DROP_OUT_SIZE,
                                 activation = ACTIVATION, 
                                 kernel_initializer= INITIALIZER, 
                                 padding= PADDING,
                                 pooling_size = POOLING_SIZE,
                                 l = L):
        
        """Creates a U-Net contraction layer consist of two convolutions, one dropout and one pooling with regularization

        Args:
            prev_contr_layer (Tensofrlow Layer): The previous layer
            filters (int): [description]
            kernel_size (array): Kernel Size
            strides (array, optional): Strides. (2,2).
            drop_out_size (float, optional): . the percentatge of droupped out connections Defaults to 0.1
            activation (string, optional): The activation function. Defaults to 'relu'.
            kernel_initializer (string, optional): The kernel intitializer. Defaults to 'he_normal''.
            padding (string, optional): convolution padding. Defaults to 'same'.
            pooling_size (array, optional): Pooling window size. Defaults to (2,2).
            l (float): the regularization strength. Default to 0.1.

        
        Returns:
            CNN U-Net contraction Layer
        """    
        
        ## Make the first convloutional
        new_contr_layer = tf.keras.layers.Conv2D( filters, 
                                                 kernel_size,            
                                                 activation = activation, 
                                                 kernel_initializer=kernel_initializer, 
                                                 padding = padding,
                                                 kernel_regularizer = tf.keras.regularizers.l2( l= l)) (prev_contr_layer)
        #Drop out some connections to avoid overfitting
        new_contr_with_drop = tf.keras.layers.Dropout(drop_out_size)(new_contr_layer)
        
        ## Make the second convolution
        new_contr_layer2 = tf.keras.layers.Conv2D( filters,  kernel_size,                                                   
                                                  activation = activation, 
                                                 kernel_initializer=kernel_initializer, 
                                                 padding = padding,
                                                 kernel_regularizer = tf.keras.regularizers.l2( l= l))(new_contr_with_drop)
       
        new_contr_layer_pooled = tf.keras.layers.MaxPooling2D(pooling_size)(new_contr_layer2)
        
        
        return new_contr_layer_pooled, new_contr_layer2
      
        
    def create_expansive_layer_regularized(self,contr_layer, prev_layer, 
                                           filters, 
                                           kernel_size, 
                                           strides, 
                                           drop_out_size = DROP_OUT_SIZE,
                                           activation = ACTIVATION, 
                                           kernel_initializer= INITIALIZER, 
                                           padding= PADDING,
                                           pooling_size = POOLING_SIZE,
                                           l = L):

        """Creates a U-Net expansion layer consist of transpose, two convolutions, one dropout with regularization

        Args:
            prev_contr_layer (Tensofrlow Layer): The corresponding previous layer
            prev_layer(Tensorflow Layer): The previous expansion Layer
            filters (int): [description]
            kernel_size (array): Kernel Size
            strides (array, optional): Strides. (2,2).
            drop_out_size (float, optional): . the percentatge of droupped out connections Defaults to 0.1
            activation (string, optional): The activation function. Defaults to 'relu'.
            kernel_initializer (string, optional): The kernel intitializer. Defaults to 'he_normal''.
            padding (string, optional): convolution padding. Defaults to 'same'.
            pooling_size (array, optional): Pooling window size. Defaults to POOLING_SIZE.
            l (float): the regularization strength. Default to 0.1.
        
        Returns:
            CNN U-Net expansion Layer
        """       

        transposed_layer = tf.keras.layers.Conv2DTranspose( filters, 
                                                           (2,2), 
                                                           strides = strides,                                                            
                                                           kernel_initializer = kernel_initializer,
                                                           padding = padding,
                                                           kernel_regularizer = tf.keras.regularizers.l2( l= l))(prev_layer)
  
        conc_layer = tf.keras.layers.concatenate([transposed_layer, contr_layer])
        
        conv1_layer = tf.keras.layers.Conv2D(filters, 
                                            kernel_size, 
                                            activation= activation,
                                            kernel_initializer = kernel_initializer,
                                            padding = padding,
                                            kernel_regularizer = tf.keras.regularizers.l2( l= l))(conc_layer)
        
        dropped_layer = tf.keras.layers.Dropout(drop_out_size)(conv1_layer)
        
        conv2_layer = tf.keras.layers.Conv2D(filters, 
                                            kernel_size, 
                                            activation=activation,
                                            kernel_initializer= kernel_initializer,
                                            padding = padding,
                                            kernel_regularizer = tf.keras.regularizers.l2( l= l))(dropped_layer)
        
        return conv2_layer 
    
    def construct_model(self,num_of_layers, start_filter, kernel_size = KERNEL_SIZE ):
        """Build a U-net model consist of contraction path and expansive path

        Args:
            num_of_layers (int): Number of needed layers
            start_filter (int): the starting filter
            kernel_size (array, optional): convloution kernel size. Defaults to KERNEL_SIZE.

        Returns:
            U-net model: a u-net model ready for training
        """        
        current_filter_size = start_filter
        input_layer = tf.keras.layers.Input(self.input_size)
        contraction_layers = []
        
        pooled_layer, unpooled_layer = self.create_contraction_layer(input_layer, 
                                              filters = current_filter_size, 
                                              kernel_size =  KERNEL_SIZE)
        #Build the contraction path 
        contraction_layers.append(unpooled_layer)
        #pooled_layer = first_layer
        unpooled_layer = None
        
        
        for contr_index in range(1,num_of_layers):
            current_filter_size = current_filter_size * 2
            pooled_layer, unpooled_layer = self.create_contraction_layer(pooled_layer, 
                                                       filters = current_filter_size, 
                                                       kernel_size =  KERNEL_SIZE)
            #print(pooled_layer.shape, unpooled_layer.shape)
            contraction_layers.append(unpooled_layer) 
            #print(next_layer.shape)
            
        current_filter_size = current_filter_size * 2   
        
        _, last_contraction_layer = self.create_contraction_layer(pooled_layer, 
                                                                  filters = current_filter_size,
                                                                  kernel_size =  KERNEL_SIZE)
        
         #Build the expansion path           
        current_filter_size = current_filter_size // 2
        
        current_exp_layer = self.create_expansive_layer(contr_layer =contraction_layers[-1] , 
                                                      prev_layer = last_contraction_layer ,
                                                      filters = current_filter_size,
                                                      kernel_size = KERNEL_SIZE,
                                                      strides =STRIDES )
        
        for exp_index in range(num_of_layers - 2, -1, -1):
            current_filter_size = current_filter_size // 2
            current_exp_layer =  self.create_expansive_layer(contr_layer = contraction_layers[exp_index], 
                                                             prev_layer =  current_exp_layer,
                                                             filters = current_filter_size,
                                                             kernel_size = KERNEL_SIZE,
                                                             strides =STRIDES  )
            
        output_filter_size = (1,1)
        output_activation = 'softmax'
        output_layer = tf.keras.layers.Conv2D( self.num_of_classes,  
                                              output_filter_size,  
                                              activation=output_activation)(current_exp_layer)
        #Compile the model and return it
        initial_learning_rate = 0.0001
        loss = 'categorical_crossentropy'
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate = initial_learning_rate)
        self.model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
        self.model.compile(optimizer=adam_optimizer, 
                           loss= loss, metrics=[tf.keras.metrics.MeanIoU(num_classes=self.num_of_classes)])
        
        return self.model

    def construct_regularized_model(self,num_of_layers, start_filter, kernel_size = KERNEL_SIZE ):
        """Build a U-net model consist of contraction path and expansive path with regularization

        Args:
            num_of_layers (int): Number of needed layers
            start_filter (int): the starting filter
            kernel_size (array, optional): convloution kernel size. Defaults to KERNEL_SIZE.

        Returns:
            U-net model: a u-net model ready for training
        """        
        current_filter_size = start_filter
        input_layer = tf.keras.layers.Input(self.input_size)
        contraction_layers = []
        #Building the contraction path
        pooled_layer, unpooled_layer = self.create_contraction_layer_regularized(input_layer,                                                                                  
                                              filters = current_filter_size, 
                                              kernel_size =  KERNEL_SIZE)
        
        contraction_layers.append(unpooled_layer)
        #pooled_layer = first_layer
        unpooled_layer = None
        
        
        for contr_index in range(1,num_of_layers):
            current_filter_size = current_filter_size * 2
            pooled_layer, unpooled_layer = self.create_contraction_layer_regularized(pooled_layer, 
                                                       filters = current_filter_size, 
                                                       kernel_size =  KERNEL_SIZE)
            
            contraction_layers.append(unpooled_layer) 
            
            
        current_filter_size = current_filter_size * 2   
        
        _, last_contraction_layer = self.create_contraction_layer_regularized(pooled_layer, 
                                                                  filters = current_filter_size,
                                                                  kernel_size =  KERNEL_SIZE)
        
        #Building the expansion path            
        current_filter_size = current_filter_size // 2
        
        current_exp_layer = self.create_expansive_layer_regularized(contr_layer =contraction_layers[-1] , 
                                                      prev_layer = last_contraction_layer ,
                                                      filters = current_filter_size,
                                                      kernel_size = KERNEL_SIZE,
                                                      strides =STRIDES )
        
        for exp_index in range(num_of_layers - 2, -1, -1):
            #print(exp_index)
            current_filter_size = current_filter_size // 2
            current_exp_layer =  self.create_expansive_layer_regularized(contr_layer = contraction_layers[exp_index], 
                                                             prev_layer =  current_exp_layer,
                                                             filters = current_filter_size,
                                                             kernel_size = KERNEL_SIZE,
                                                             strides =STRIDES  )
        # Add the activation layer            
        output_filter_size = (1,1)
        output_activation = 'softmax'
        output_layer = tf.keras.layers.Conv2D( self.num_of_classes,  output_filter_size,  activation=output_activation)(current_exp_layer)
        #Compile the model ard return it
        initial_learning_rate = 0.0001
        loss = 'categorical_crossentropy'
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate = initial_learning_rate)
        self.model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
        self.model.compile(optimizer=adam_optimizer, 
                           loss= loss, 
                           metrics=[tf.keras.metrics.MeanIoU(num_classes=self.num_of_classes)])
        
        return self.model
        
        

from keras.layers import SimpleRNN
import keras.backend as K

class NoInputRNN(SimpleRNN):
    def __init__(self, *kargs, **kwargs):
        if 'return_sequences' in kwargs:
            raise Exception('must be true')
        self.num_timesteps = kwargs['num_timesteps']
        del kwargs['num_timesteps']
        super(NoInputRNN, self).__init__(*kargs, **kwargs)
        self.return_sequences = True
        
    def build(self, input_shape):
        super(NoInputRNN, self).build(input_shape)
        self.trainable_weights = [self.U, self.b]
        
    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], self.num_timesteps, self.output_dim)
        else:
            return (input_shape[0], self.output_dim)
        
    def step(self, x, states): # ignores input
        prev_output = states[0]
        B_U = states[1]
        output = self.activation(K.dot(prev_output * B_U, self.U) + self.b)
        return output, [output]
    
    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        
        inputs = K.repeat_elements(x * 0, self.num_timesteps, 1)
        
        initial_states = [x[:,0,:],]
        constants = self.get_constants(x)
        #preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, inputs,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=self.num_timesteps)
        if self.return_sequences:
            return outputs
        else:
            return last_output

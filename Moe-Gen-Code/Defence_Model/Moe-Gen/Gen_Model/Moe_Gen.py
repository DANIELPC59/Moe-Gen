import tensorflow as tf
from tensorflow.keras import layers, Model


class MoE_Gen(tf.keras.Model):
    def __init__(self, noise_size, flow_type, output_size, num_experts=8, embed_dim=64, dropout_rate=0.05,temperature=0.05):
        super(MoE_Gen, self).__init__()
        self.noise_size = noise_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.temperature=temperature
        
        self.class_embed = layers.Embedding(input_dim=flow_type, output_dim=embed_dim)

        # Gate
        self.gate_dense = layers.Dense(num_experts)
        self.softmax = layers.Softmax(axis=-1)

        # Gen
        self.experts = [self._build_expert(noise_size + embed_dim, dropout_rate) for _ in range(num_experts)]

    def _build_expert(self, input_dim, dropout_rate):
        return tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(512, activation='relu'),
        ])

    def call(self, batch_size, batch_label, training=False):
        embed_y = self.class_embed(batch_label)  # [B, embed_dim]

        # Gata
        gate_logits = self.gate_dense(embed_y)  # [B, num_experts]
        # print(gate_logits[:5])
        gate_weights = tf.nn.softmax(gate_logits / self.temperature, axis=-1)  

        # random noise
        z = tf.random.normal(shape=(batch_size, self.noise_size))
        x_input = tf.concat([z, embed_y], axis=-1)  # [B, noise + embed_dim]

        # get perpubate
        expert_outputs = tf.stack([expert(x_input, training=training) for expert in self.experts], axis=1)  # [B, num_experts, 512]

        # combine
        gate_weights = tf.expand_dims(gate_weights, -1)  # [B, num_experts, 1]
        x = tf.reduce_sum(gate_weights * expert_outputs, axis=1)  # [B, 512]

        # mask 
        mask = tf.cast(tf.range(tf.shape(x)[1]) % 2 == 0, tf.float32)
        x = x * mask

        # clip
        feature_dim = tf.shape(x)[1]
        if feature_dim < self.output_size:
            pad_size = self.output_size - feature_dim
            padding = tf.zeros([batch_size, pad_size], dtype=x.dtype)
            x = tf.concat([x, padding], axis=1)
        else:
            x = x[:, :self.output_size]
        x=x[:,:,tf.newaxis]
        if training:
            return x, gate_weights
        else:
            return x
    
    
    
   

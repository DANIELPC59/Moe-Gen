def burst_to_direction(burst_seq, max_dir_len=5000):
    """
    Another optimized implementation: use tf.repeat for vectorized processing
    Process each sample individually, but use vectorized operations internally
    """
    """
    Fixed version: solve the problem that mask cannot be a scalar
    Ensure mask dimension is correct when using tf.boolean_mask
    """
    batch_size = tf.shape(burst_seq)[0]
    result = []
    
    for i in tqdm(tf.range(batch_size)):
        bursts = tf.squeeze(burst_seq[i])  
        
        non_zero_mask = tf.not_equal(bursts, 0)
        non_zero_mask = tf.reshape(non_zero_mask, [-1])  
        
        non_zero_bursts = tf.boolean_mask(bursts, non_zero_mask)
        
        dirs = tf.sign(non_zero_bursts)
        counts = tf.abs(non_zero_bursts)
        
        dir_sequence = tf.repeat(dirs, counts)
        
        current_len = tf.shape(dir_sequence)[0]
        if current_len > max_dir_len:
            dir_sequence = dir_sequence[:max_dir_len]
        else:
            pad_len = max_dir_len - current_len
            dir_sequence = tf.pad(dir_sequence, [[0, pad_len]])
        
        result.append(dir_sequence)
    
    return tf.stack(result)



def direction_to_burst(packet_seq,fill_burst=False):
    if not np.any(packet_seq):
        return []

    burst = []
    current_burst_count = 0
    current_burst_direction = np.sign(packet_seq[0])

    for i, value in enumerate(packet_seq):
        if value == 0:
            if current_burst_count > 0:
                burst.append(current_burst_direction * current_burst_count)
                current_burst_count =0
            break
        else:
            if np.sign(value) == current_burst_direction:
                current_burst_count += 1
            else:
                burst.append(current_burst_direction * current_burst_count)
                current_burst_direction = np.sign(value)
                current_burst_count = 1

    if current_burst_count > 0:
        burst.append(current_burst_direction * current_burst_count)
    if(fill_burst):
        burst_length = len(burst)
        if(burst_length>2000):
            burst=burst[:2000]
        elif(burst_length<2000):
            burst.extend([0]*(2000-burst_length))
            
    return burst

import tensorflow as tf

num_features = 33762578

g = tf.Graph()
records = [["00", "01", "02", "03", "04"], ["05", "06", "07", "08", "09"], ["10", "11", "12", "13", "14"],
           ["15", "16", "17", "18", "19"], ["20", "21"]]

with g.as_default():
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.ones([33762578]), name="model")

    gradients = []
    # For every VM use the respective tfrecords and calculate the sparse gradient, and append it to gradients
    for i in range(0, 5):
        with tf.device("/job:worker/task:%d" % i):

            filename_queue = tf.train.string_input_producer([
                "/home/ubuntu/tfrecords/tfrecords" + record for record in
                records[i]], num_epochs=None)

            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'label': tf.FixedLenFeature([1], dtype=tf.int64),
                    'index': tf.VarLenFeature(dtype=tf.int64),
                    'value': tf.VarLenFeature(dtype=tf.float32),
                }
            )
            label = features['label']
            index = features['index']
            value = features['value']

            local_gradient_sparse = tf.SparseTensor(shape=[num_features],
                                                    indices=[features['index'].values],
                                                    values=tf.mul(tf.mul(tf.cast(label, tf.float32), tf.mul(
                                                        tf.sigmoid(tf.mul(tf.cast(label, tf.float32), tf.reduce_sum(
                                                            tf.mul(tf.gather(w, features['index'].values),
                                                                   features['value'].values)))) - 1,
                                                        features['value'].values)), -0.01))

            gradients.append(local_gradient_sparse)

    # Collect the gradients tp form the model and calculate the error on the test sample
    with tf.device("/job:worker/task:0"):
        temp_gradient = gradients[0]

        i = 0
        while i in range(1, len(gradients)):
            temp_gradient = tf.sparse_add(temp_gradient, gradients[i])

        assign_op = tf.scatter_add(w, tf.reshape(temp_gradient.indices, [-1]), tf.reshape(temp_gradient.values, [-1]))

        filename_queue2 = tf.train.string_input_producer(
            ["/home/ubuntu/tfrecords/tfrecords22"], num_epochs=None)
        _, data_serialized = tf.TFRecordReader().read(filename_queue2)
        feature = tf.parse_single_example(data_serialized,
                                          features={
                                              'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                              'index': tf.VarLenFeature(dtype=tf.int64),
                                              'value': tf.VarLenFeature(dtype=tf.float32),
                                          })
        label = features['label']
        index = features['index']
        value = features['value']

        feature_dense = tf.sparse_to_dense(sparse_indices=tf.sparse_tensor_to_dense(index),
                                           output_shape=[num_features, ],
                                           sparse_values=tf.sparse_tensor_to_dense(value))

        y_hat = tf.reduce_sum(tf.mul(w, feature_dense))
        test_error = tf.mul(tf.cast(
            tf.sub(tf.constant([1], dtype=tf.int64),
                   tf.mul(tf.cast(tf.sign(y_hat), tf.int64), features['label'])),
            tf.float32), 0.5)

    # For every VM run 10000 iterations and afer every 100 iteration calculate the test error using 2000 data points
    # Collect the errors in a file called async_errors
    with tf.Session("grpc://vm-22-1:2222") as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())
        tf.train.start_queue_runners(sess=sess)
        for i in range(0, 10000):
            run_cross_validation = i != 0 and i % 100 == 0
            if run_cross_validation:
                total_error_rate = 0
                for j in range(0, 2000):
                    sess.run(test_error)
                    total_error_rate += test_error.eval()[0]
                total_error_rate = (total_error_rate / 2000) * 100
                with open("sync_errors", "a+") as syns_errors:
                    syns_errors.write(
                        "\nIteration- " + str(i) + " Error Rate Percentage: " + str(total_error_rate))
            sess.run(assign_op)
            print(w.eval())

        sess.close()
import tensorflow as tf  
import pdb
 
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
t3 = [[3,6,9]]

z1 = tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
z2 = tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
z3 = tf.add(t1,t2)
 
with tf.Session() as sess:  
    print(sess.run(z1))
    
pdb.set_trace()
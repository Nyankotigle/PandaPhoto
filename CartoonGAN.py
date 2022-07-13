import numpy as np
import cv2
import tensorflow as tf
import panda_shot.network as network
import panda_shot.guided_filter as guided_filter

def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    print("CartoonGAN__resize__start!!!")
    image = cv2.resize(image, (w, h),interpolation=cv2.INTER_AREA)
    print("CartoonGAN__resize__end!!!")
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image



def CartoonGAN(frame):

    print("CartoonGAN")
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    print("1")
    network_out = network.unet_generator(input_photo)
    print("2")
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)
    print("3")


    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('panda_shot/CartoonGAN/saved_models'))
    print("4")
    image = resize_crop(frame)
    batch_image = image.astype(np.float32) / 127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    output = sess.run(final_out, feed_dict={input_photo: batch_image})
    output = (np.squeeze(output) + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output
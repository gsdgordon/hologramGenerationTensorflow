# A minimum working example of how to generate multiphase phase-only holograms using the ADAM optimiser in TensorFlow
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as spio
import tensorflow as tf
import h5py
import time


def build_initializers():
    # Load the target image
    #mat = spio.loadmat('target.mat', squeeze_me=True)
    #targetMat = mat['A_mat_orig1']
    myImage = Image.open("ideaUSAF.jpg")
    myImage.load()
    data3d = np.array(myImage, dtype="complex128")
    data2d = data3d[:,:,1]

    imDims = data2d.shape

    #If you want to view the target
    #plt.imshow(np.absolute(data2d))
    #plt.show()

    # Define a 'region of interest' to enable improve convergence if optimising both amplitude and phase
    zeroMargin = 50
    ROI = np.ones(imDims)
    ROI[1:zeroMargin, :] = 0
    ROI[:, 1:zeroMargin] = 0
    ROI[-zeroMargin:-1,:] = 0
    ROI[:, -zeroMargin:-1] = 0

    #plt.imshow(np.absolute(ROI))
    #plt.show()

    return imDims, data2d, ROI

def build_graph(target, startHolo, ROI, imDims):
    graph = tf.Graph()
    ops = dict()

    with graph.as_default():
        target_tf_r = tf.Variable(target.real, trainable=False, name='target_tf_r')
        target_tf_i = tf.Variable(target.imag, trainable=False, name='target_tf_i')
        target_tf = tf.complex(target_tf_r, target_tf_i, 'target_tf')

        ROI_tf = tf.Variable(ROI, trainable=False, name='ROI_tf')
        #ROI_i = tf.Variable(np.zeros(imDims), trainable=False, name='ROI_tf_i')
        #ROI_tf = tf.complex(target_tf_r, target_tf_i, 'ROI_tf')

        #initHolo = np.zeros((imDims(1),imDims(2)))
        #holo_tf_r = tf.Variable(startHolo.real, name='holo_tf_r')
        #holo_tf_i = tf.Variable(startHolo.imag, name='holo_tf_i')
        #holo_tf = tf.complex(holo_tf_r, holo_tf_i, 'holo_tf')
        #holo_ang_tf = tf.angle(holo_tf)

        #Optimise the phase values of each pixel
        holo_ang_tf = tf.Variable(startHolo, name='holo_ang_tf')
        holo_comp_tf = tf.complex(tf.cos(holo_ang_tf),tf.sin(holo_ang_tf))

        illumAmp = np.ones(imDims)
        illumAmp_tf_r = tf.Variable(illumAmp.real, trainable=False, name='illumAmp_tf_r')
        illumAmp_tf_i = tf.Variable(illumAmp.imag, trainable=False, name='illumAmp_tf_i')
        illumAmp_tf = tf.complex(illumAmp_tf_r, illumAmp_tf_i, 'illumAmp_tf')

        holoField_tf = tf.multiply(holo_comp_tf,illumAmp_tf)

        shiftAmount_0 = np.int_(imDims[0]/2)
        shiftAmount_1 = np.int_(imDims[1]/2)
        holoField_tf_shift = tf.manip.roll(holoField_tf, shift=tf.constant(shiftAmount_0,dtype='int32'),axis=0)
        holoField_tf_shift = tf.manip.roll(holoField_tf_shift, shift=tf.constant(shiftAmount_1,dtype='int32'), axis=1)
        replayField_tf_shift = tf.fft2d(holoField_tf_shift)

        replayField_tf = tf.manip.roll(replayField_tf_shift, shift=tf.constant(shiftAmount_0,dtype='int32'),axis=0)
        replayField_tf = tf.manip.roll(replayField_tf, shift=tf.constant(shiftAmount_1,dtype='int32'), axis=1)

        replayField_ROI_tf =  tf.multiply(replayField_tf,tf.complex(ROI_tf,np.zeros(imDims)))
        target_ROI_tf = tf.multiply(target_tf,tf.complex(ROI_tf,np.zeros(imDims)))

        replayField_abs_tf = tf.abs(replayField_ROI_tf)
        rf_norm = tf.norm(replayField_abs_tf, ord='fro', axis=(0, 1))
        targ_norm = tf.norm(tf.abs(target_ROI_tf), ord='fro', axis=(0,1))
        replayField_ROI_norm_tf = tf.divide(replayField_ROI_tf,tf.complex(rf_norm,tf.constant(0,dtype='float64')))
        target_ROI_norm_tf = tf.divide(target_ROI_tf, tf.complex(targ_norm, tf.constant(0,dtype='float64')))

        # Compute the error function.  Can modify to only optimise amplitude of replay field
        errMat = tf.abs(replayField_ROI_norm_tf - target_ROI_norm_tf)
        #errMat = tf.abs(tf.abs(replayField_norm_tf) - tf.abs(target_norm_tf))
        errMat = tf.multiply(errMat,ROI_tf)

        pow_ROI = tf.pow(tf.norm(tf.abs(replayField_ROI_tf), ord='fro', axis=(0, 1)),2)
        pow_full = tf.pow(tf.norm(tf.abs(replayField_tf), ord='fro', axis=(0, 1)),2)
        efficiency = tf.divide(pow_ROI, pow_full)*100

        loss_functions = 1000*tf.sqrt(tf.reduce_sum(tf.square(errMat)))

        train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_functions)
        #train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss_functions)

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        ops['bestHoloAng'] = holo_ang_tf
        ops['train'] = train
        ops['init'] = init
        ops['loss_functions'] = loss_functions
        ops['replayField_abs'] = replayField_abs_tf
        ops['efficiency'] = efficiency

        return graph, ops

def run_graph(i_max, graph, ops):
    with tf.Session(graph=graph) as sess:
        #writer = tf.summary.FileWriter('./graphs', sess.graph)
        sess.run(ops['init'])
        #print_matrices_functions_decompositions(sess, ops)

        plt.ion()
        plt.show()
        fig = plt.figure()

        for i in range(i_max):
            errFun, efficiency, replayField_abs, _ = sess.run([ops['loss_functions'],ops['efficiency'],ops['replayField_abs'], ops['train']])
            print('i={}, errFun  {:.4f}, efficiency {:.4f}%'.format(i, errFun, efficiency))

            plt.imshow(np.log10(replayField_abs))
            #plt.draw()
            fig.canvas.flush_events()
            #plt.pause(0.001)
            time.sleep(0.001)

            if i % 10 == 0:
                bestHolo = sess.run([ops['bestHoloAng']])
                spio.savemat('bestHolo_temp.mat', mdict={'holo': bestHolo, 'iter': i})


            if i == (i_max - 1):
            #     save_matrix_mat(sess, ops)
            #    print_matrices_functions_decompositions(sess, ops)
                bestHolo = sess.run([ops['bestHoloAng']])

                spio.savemat('bestHolo.mat', mdict={'holo': bestHolo})
                return bestHolo


def test():
    imDims, target, ROI = build_initializers()

    #Define starting hologram
    #startHolo_ang = np.ones(imDims)
    startHolo_ang = np.random.rand(imDims[0],imDims[1])*2*np.pi

    #mat = spio.loadmat('bestHolo_temp.mat', squeeze_me=True)
    #startHolo_ang = mat['holo']

    # Set up the problem
    graph, ops = build_graph(target, startHolo_ang, ROI, imDims)

    i_max = 2000
    run_graph(i_max, graph, ops)


if __name__ == '__main__':
    test()
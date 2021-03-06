import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors

DATAPATHS = ['data/AR_FaceDB_haarNose_train/', 'data/AR_FaceDB_haarNose_test/']


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            np.random.seed(seed=args.seed)
            
            datasetTrain = facenet.get_dataset(DATAPATHS[0])
            datasetTest = facenet.get_dataset(DATAPATHS[1])
            pathsTrain, labelsTrain = facenet.get_image_paths_and_labels(datasetTrain)
            pathsTest, labelsTest = facenet.get_image_paths_and_labels(datasetTest)
            print('Number of classes train test: ', len(datasetTrain), len(datasetTest))
            print('Number of images train test: ', len(pathsTrain), len(labelsTrain), len(pathsTest), len(labelsTest))
            

            # Load the model
            facenet.load_model(args.model)
            

            # DO TRAINING 
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            nrof_images = len(pathsTrain)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = pathsTrain[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            # Train classifier
            # gridSearchParams = {'estimator__C': [0.0001, 0.001, 0.01, 0.1, 1]}
            # model = GridSearchCV(OneVsRestClassifier(SVC(kernel='linear', probability=True)), gridSearchParams)

            model = SVC(kernel='linear', probability=True)

            # model = MLPClassifier(solver='adam', activation='relu', max_iter=200, alpha=1e-5, hidden_layer_sizes=(64), random_state=1)

            # model =  NearestNeighbors(n_neighbors=10)
            # model.fit(emb_array)
            
            model.fit(emb_array, labelsTrain)
            # print(model.best_params_)


            # Create a list of class names
            class_names = [ cls.name.replace('_', ' ') for cls in datasetTrain]

            
            # DO TESTING 
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            nrof_images = len(pathsTest)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = pathsTest[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            # Classify images
            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            
            for i in range(len(best_class_indices)):
                print('%s, %s, %f' % (class_names[labelsTest[i]], class_names[best_class_indices[i]], best_class_probabilities[i]))
                
            for x in predictions:
                print(list(x))

            accuracy = np.mean(np.equal(best_class_indices, labelsTest))
            print('Accuracy: %.3f' % accuracy)
                

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch.', default=64)
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int, help='Random seed.', default=666)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

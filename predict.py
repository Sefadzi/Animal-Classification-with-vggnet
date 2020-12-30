from keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image we are going to classify")
ap.add_argument("-m", "--model", required=True,
                help="path to output trained keras model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
ap.add_argument("-w", "--width", type=int, default=28,
                help="target spatial dimension width")
ap.add_argument("-e", "--height", type=int, default=28,
                help="target spatial dimension height")
ap.add_argument("-f", "--flatten", type=int, default=-1,
                help="whether or no we should flatten the image")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
output = np.copy(image)
image = cv2.resize(image, (args["width"], args["height"]))

# flatten image if necessary
# check to see if we should flatten image and add batch
# dimension

if args["flatten"] > 0:
    image = image.flatten()
    image = image.reshape((1, image.shape[0]))
    # otherwise we must be working with a CNN -- don't flatten the
    # #image, simply add the batch dimension
else:
    image = image.reshape((1, image.shape[0], image.shape[1],
                           image.shape[2]))


# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# make a prediction on the same image
preds = model.predict(image)

# find the class label index with the largest corresponding probability
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# draw the class label + probability on the output image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX,
            0.7, (0, 0, 255), 2)

# SHOW THE OUTPUT IMAGE
cv2.imshow("Image", output)
cv2.waitKey(0)








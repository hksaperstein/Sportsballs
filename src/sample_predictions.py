import cv2 as cv
# import src.three_classes_unreg as model
# import src.three_classes_reg as model
# import src.seven_classes_unreg as model
import src.seven_classes_reg as model

'''
Prints sample prediction for one of each class
# load image from file
# resize image for network
# print class and prediction to console
# OpenCV display image
'''
# TODO: un-one-hot-encode outputs
model = model.load_weights()

# Baseball sample
im = 'data/baseball/n02799071_355.JPEG'
im = cv.imread(im)
baseball = cv.resize(im, dsize=(128,128))
print('Baseball (1):')
print(model.predict(baseball.reshape(1,128,128,3)))
# cv.imshow('baseball', baseball)
# cv.waitKey(0)

# Basketball sample
im = 'data/basketball/n02802426_22.JPEG'
im = cv.imread(im)
basketball = cv.resize(im, dsize=(128,128))
print('Basketball (2):')
print(model.predict(basketball.reshape(1,128,128,3)))
# cv.imshow('basketball', basketball)
# cv.waitKey(0)

# Golf sample
im = 'data/golf balls/n03445777_176.JPEG'
im = cv.imread(im)
golf = cv.resize(im, dsize=(128,128))
print('Golf ball (3):')
print(model.predict(golf.reshape(1,128,128,3)))
# cv.imshow('golf', golf)
# cv.waitKey(0)

# Not sportsballs sample
im = 'data/not sportsballs/not sportsballs0-frame385.jpg'
im = cv.imread(im)
ns = cv.resize(im, dsize=(128,128))
print('Not sportsball (4):')
print(model.predict(ns.reshape(1,128,128,3)))
# cv.imshow('ns', ns)
# cv.waitKey(0)

# Soccer sample
im = 'data/soccer ball/n04254680_74.JPEG'
im = cv.imread(im)
soccer = cv.resize(im, dsize=(128,128))
print('soccer (5):')
print(model.predict(soccer.reshape(1,128,128,3)))
# cv.imshow('soccer', soccer)
# cv.waitKey(0)

# Tennis sample
im = 'data/tennis balls/n04409515_265.JPEG'
im = cv.imread(im)
Tennis = cv.resize(im, dsize=(128,128))
print('Tennis (6):')
print(model.predict(Tennis.reshape(1,128,128,3)))
# cv.imshow('Tennis', Tennis)
# cv.waitKey(0)

# Volleyball sample
im = 'data/volleyballs/n04540053_222.JPEG'
im = cv.imread(im)
volleyballs = cv.resize(im, dsize=(128,128))
print('volleyballs (7):')
print(model.predict(volleyballs.reshape(1,128,128,3)))
# cv.imshow('volleyballs', volleyballs)
# cv.waitKey(0)





# # Predict for custom image:
# im = 'kristen.png'
# im = cv.imread(im)
# kristen = cv.resize(im, dsize=(128,128))
# # cv.imshow('golf', golf)
# # cv.waitKey(3000)
# print('Not sportsball (4):')
#
# print(model.predict(golf.reshape(1,128,128,3)))
#
# # Predict for custom image:
# im = 'tennis2.png'
# im = cv.imread(im)
# tennis2 = cv.resize(im, dsize=(128,128))
# # cv.imshow('golf', golf)
# # cv.waitKey(3000)
# print('Tennis ball (4):')
#
# print(model.predict(golf.reshape(1,128,128,3)))
cv.waitKey(0)
cv.destroyAllWindows()

# Incorrect predictions:
im = 'data/not sportsballs/not sportsballs0-frame56.jpg'

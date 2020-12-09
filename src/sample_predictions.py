import cv2 as cv
# import src.three_classes_unreg as model
# import src.three_classes_reg as model
# import src.seven_classes_unreg as model
import src.seven_classes_reg as model


# TODO: un-one-hot-encode outputs

model = model.load_weights()

# Baseball sample
im = 'data/baseball/n02799071_355.JPEG'
im = cv.imread(im)
baseball = cv.resize(im, dsize=(128,128))
# cv.imshow('baseball', baseball)
# cv.waitKey(3000)
print('Baseball(1):')

print(model.predict(baseball.reshape(1,128,128,3)))

# Basketball sample
im = 'data/basketball/n02802426_22.JPEG'
im = cv.imread(im)
basketball = cv.resize(im, dsize=(128,128))
# cv.imshow('basketball', basketball)
# cv.waitKey(3000)
print('Basketball (2):')
print(model.predict(basketball.reshape(1,128,128,3)))

# Golf sample
im = 'data/golf balls/n03445777_176.JPEG'
im = cv.imread(im)
golf = cv.resize(im, dsize=(128,128))
# cv.imshow('golf', golf)
# cv.waitKey(3000)
print('Golf ball (3):')
print(model.predict(golf.reshape(1,128,128,3)))

# Not sportsballs sample
im = 'data/not sportsballs/not sportsballs0-frame385.jpg'
im = cv.imread(im)
ns = cv.resize(im, dsize=(128,128))
cv.imshow('golf', ns)
# cv.waitKey(0)
print('Not sportsball (4):')
print(model.predict(ns.reshape(1,128,128,3)))

# Soccer sample
im = 'data/soccer ball/n04254680_74.JPEG'
im = cv.imread(im)
soccer = cv.resize(im, dsize=(128,128))
# cv.imshow('golf', golf)
# cv.waitKey(3000)
print('soccer (5):')
print(model.predict(soccer.reshape(1,128,128,3)))

# Tennis sample
im = 'data/tennis balls/n04409515_265.JPEG'
im = cv.imread(im)
Tennis = cv.resize(im, dsize=(128,128))
# cv.imshow('golf', golf)
# cv.waitKey(3000)
print('Tennis (6):')
print(model.predict(Tennis.reshape(1,128,128,3)))

# Volleyball sample
im = 'data/volleyballs/n04540053_222.JPEG'
im = cv.imread(im)
volleyballs = cv.resize(im, dsize=(128,128))
# cv.imshow('golf', golf)
# cv.waitKey(3000)
print('volleyballs (7):')
print(model.predict(volleyballs.reshape(1,128,128,3)))





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

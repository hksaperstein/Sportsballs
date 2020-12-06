import cv2 as cv
import src.three_classes_unreg as model
# import src.three_classes_reg as model
# import src.seven_classes_unreg as model
# import src.seven_classes_reg as model


# TODO: un-one-hot-encode outputs

model = model.load_weights()

im = 'data/baseball/n02799071_355.JPEG'
im = cv.imread(im)
baseball = cv.resize(im, dsize=(128,128))
# cv.imshow('baseball', baseball)
# cv.waitKey(3000)

print(model.predict(baseball.reshape(1,128,128,3)))

im = 'data/basketball/n02802426_22.JPEG'
im = cv.imread(im)
basketball = cv.resize(im, dsize=(128,128))
# cv.imshow('basketball', basketball)
# cv.waitKey(3000)

print(model.predict(basketball.reshape(1,128,128,3)))

im = 'data/golf balls/n03445777_59.JPEG'
im = cv.imread(im)
golf = cv.resize(im, dsize=(128,128))
# cv.imshow('golf', golf)
# cv.waitKey(3000)

print(model.predict(golf.reshape(1,128,128,3)))

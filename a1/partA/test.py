import cv2
import numpy as np
backB = cv2.imread("photo/2/back2.jpg")
compB = cv2.imread("photo/2/comp2.jpg")
diffB = backB - compB
diff = np.sum(np.absolute(diffB))
r = diffB[:,:,2]
g = diffB[:,:,1]
b = diffB[:,:,0]
cv2.imshow("diff", diffB)
cv2.waitKey
print 1

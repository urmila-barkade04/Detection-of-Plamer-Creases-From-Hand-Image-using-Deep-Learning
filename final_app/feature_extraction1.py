


import cv2
import rembg
import matplotlib.pyplot as plt
import numpy as np

def remove_background(input_path, output_path):
    with open(input_path, "rb") as input_file, open(output_path, "wb") as output_file:
        input_data = input_file.read()

        # Use rembg library to remove the background
        output_data = rembg.remove(input_data)

        output_file.write(output_data)
def palm_lines():
                # Example usage
                input_image_path = 'input.png'
                output_image_path = 'output.png'

                remove_background(input_image_path, output_image_path)

                


                # In[ ]:





                # In[2]:


                import cv2
                import numpy as np
                import matplotlib.pyplot as plt

                def save_image_file(img, name):
                    # Specify the desired directory structure and filename
                    file_path = f"static/images/{name}.png"

                    # Save the image to the specified path
                    cv2.imwrite(file_path, img)
                # Load the original image
                original = cv2.imread("output.png")

                # Convert to grayscale
                img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                save_image_file(img, "gray")

                # Equalize histogram
                img = cv2.equalizeHist(img)
                save_image_file(img, "equalize")

                # Gaussian blur
                img = cv2.GaussianBlur(img, (9, 9), 0)
                save_image_file(img, "blur")

                # Canny edge detection
                #img = cv2.Canny(img,30,50)
                img=cv2.Canny(img,50,80,apertureSize = 3)
                save_image_file(img, "canny")

                # Hough lines
                lined = np.copy(original) * 0
                lines = cv2.HoughLinesP(img, 1, np.pi / 180, 15, np.array([]), 50, 20)
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(lined, (x1, y1), (x2, y2), (0, 0, 255))
                save_image_file(lined, "lined")

                # Combine original and lines
                output = cv2.addWeighted(original, 0.8, lined, 1, 0)
                save_image_file(output, "output")




palm_lines()



# In[ ]:





# Deteksi Uang Koin Menggunakan OpenCV Python


## Background

Along with the advancement of digital image processing technology, various computer-based automated solutions are now widely applied in everyday life, including in recognizing physical objects such as coins. In the context of microfinance, such as in small businesses, manual parking systems, or money changers, the process of identifying the value of coins is often still done manually. This is not only time-consuming, but also prone to human error. 
The use of computer vision technology such as OpenCV provides a fast, efficient, and reliable alternative to automatically recognize and classify coin denominations. This project aims to develop a system that is able to detect and classify rupiah coins with denominations of Rp100, Rp200, Rp500, and Rp1000 using a specially trained YOLOv8 model, and integrate it with OpenCV to automatically calculate the total value and display the results visually on a webcam streaming video. 
The total value of the money is then calculated automatically and displayed in real-time.

## Progress

The process stages in this project can be explained as follows:
- Image Data Collection
Coin images of various denominations are stored in a dataset folder that has been grouped based on coin value.

- Total Value Calculation
All recognized coins will be added up to display the total value of money contained in the image.

- Result Visualization
The value of each coin will be displayed at the position of the coin, and the total money will be displayed in one frame of the final visualization results.

## Libraries Used
- cv2 → To access webcam, display video, draw detection boxes, and text.
- ultralytics → To load and run YOLOv8 model.

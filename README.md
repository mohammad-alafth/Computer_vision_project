# Deteksi Uang Koin Menggunakan OpenCV Python


## Background

Along with the advancement of digital image processing technology, various computer-based automated solutions are now widely applied in everyday life, including in recognizing physical objects such as coins. In the context of microfinance, such as in small businesses, manual parking systems, or money changers, the process of identifying the value of coins is often still done manually. This is not only time-consuming, but also prone to human error. 
The use of computer vision technology such as OpenCV provides a fast, efficient, and reliable alternative to automatically recognize and classify coin denominations. This project aims to develop a system that is able to detect and classify rupiah coins with denominations of Rp100, Rp200, Rp500, and Rp1000 using a specially trained YOLOv8 model, and integrate it with OpenCV to automatically calculate the total value and display the results visually on a webcam streaming video. 
The total value of the money is then calculated automatically and displayed in real-time.

## Objectives
- Detect and classify coin types.
- Provide accuracy (confidence) in the classification results.
- Provide information on the total number of detected coins.
- Display visual box boundaries to make it easier for users to recognize detected coins.
- Implement the system in an easy-to-use web application.

---

## 📁 Dataset
- Dataset: Coin Rupiah Detection Computer Vision Project 
- Source: [Roboflow - dataset Coin Rupiah Detection Computer Vision Project by Chandras](https://universe.roboflow.com/chandras/coin-rupiah-detection)

---

## Tools & Teknologi
- *Bahasa Pemrograman:* Python  
- *Metode:* Yolov8n
- *Visualisasi:* CV, gradio
- *Teknik:* Image preprocessing, feature extraction, classification, bounding box detection  

---

## System Features
1. *Image Upload:* Users can upload coin images.
2. *Coin Prediction:*
- The coin type will be displayed automatically on the screen.
- Displays the accuracy level.
3. *Bouncing Box Visualization:*
- The coin area is marked with bounding.
- The coin type label and confidence are displayed in the image.
4. *Summing detected coins:*
- For example: "There are 5 coins of 200, then the result will be 1000."
5. *Simple Web Interface:*
- Displays the classification results, confidence, original image + bounding box, and handling.

---

## Progress Details
- ✅ Data preprocessing and image augmentation completed.
- ✅ Yolo model trained with validation accuracy up to *92%*.
- ✅ Implementation of bounding box CV completed.
- ✅ System outputs classification, confidence, number of coins, and result image.

## Results 
- ### Grafik akurasi dan loss model
  <p align="center">
   <img src="P_curve.png" width="400"/>
   <img src="results.png" width="500"/>
  </p>
 
- ### Confusion Matrix
  <p align="center">
  <img src="confusion_matrix.png" width="350"/>
  </p>


  ---
## Libraries Used
- cv2 → To access webcam, display video, draw detection boxes, and text.
- ultralytics → To load and run YOLOv8 model.

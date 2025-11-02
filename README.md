# FFR-GAN

This project explores how deep learning can improve the accuracy of **Fractional Flow Reserve (FFR)** predictions used in assessing coronary artery disease.  
The goal is to build a **conditional GAN (Generative Adversarial Network)** that adjusts FFR values based on patient-specific data such as age, sex, and health conditions.  
By doing this, the model aims to make FFR predictions more realistic and personalised.

---

## Project Overview
A baseline **Convolutional Neural Network (CNN)** is used to predict FFR values.  
The GAN then refines these predictions by learning the relationship between patient factors and variations in coronary flow.  
The system will be tested to see if the adjusted values show improved accuracy and consistency compared to the baseline.

---

## Tools and Libraries
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- scikit-learn  
- Google Colab for training and testing  

---

## Expected Results
The final outcome is a simple, working prototype that shows how conditioning FFR estimation on real patient data can lead to more personalised results.  
It will also demonstrate how generative models can be used in a medical research setting.

---

## Author
**Ramah Almazroui**  
University of Birmingham  
GitHub: [@rammazroui123](https://github.com/rammazroui123)
